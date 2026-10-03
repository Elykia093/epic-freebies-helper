import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from hcaptcha_challenger.models import ImageAreaSelectChallenge

from extensions.llm_adapter import _GLMAsyncModels
from extensions.llm_errors import LLMRequestAbort


def _client():
    return _GLMAsyncModels(
        settings=SimpleNamespace(
            GLM_BASE_URL="https://example.invalid/v1",
            GLM_API_KEY=SimpleNamespace(get_secret_value=lambda: "not-a-real-key"),
            GLM_REQUEST_TIMEOUT_SECONDS=50,
        ),
        storage={},
    )


def _config():
    return SimpleNamespace(
        response_schema=ImageAreaSelectChallenge,
        system_instruction="Select the matching tiles.",
        temperature=0.1,
        thinking_config=None,
    )


def _contents():
    return SimpleNamespace(
        role="user",
        parts=[
            SimpleNamespace(text="Find the matching tile in this image."),
            SimpleNamespace(inline_data=SimpleNamespace(data=b"image", mime_type="image/png")),
        ],
    )


def _success():
    return {
        "choices": [
            {
                "message": {
                    "content": '{"answer":[[10,20,30,60]]}',
                    "reasoning_content": '{"answer":[[100,200,300,600]]}',
                }
            }
        ]
    }


def _mock_endpoint(monkeypatch, replies):
    requests = []
    real_async_client = httpx.AsyncClient

    def handle(request):
        # Decode now so a later mutation of the original payload cannot hide a regression.
        requests.append(json.loads(request.content))
        assert len(requests) <= len(replies), "Unexpected extra API request"
        reply = replies[len(requests) - 1]
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        return httpx.Response(status, json=body)

    def make_client(**kwargs):
        return real_async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return requests


def _generate(client, model="zai-org/GLM-4.5V"):
    return asyncio.run(client.generate_content(model, _contents(), config=_config()))


@pytest.mark.parametrize(
    "error",
    [
        {"code": 20024, "message": "JSON mode is not supported"},
        {"error": {"code": "20024", "message": "Unsupported response format"}},
        {"error": {"message": "JSON MODE IS NOT SUPPORTED for this model"}},
    ],
    ids=["flat-code", "nested-code", "nested-message"],
)
def test_json_mode_fallback_preserves_images_schema_and_answer_parsing(monkeypatch, error):
    requests = _mock_endpoint(monkeypatch, [(400, error), (200, _success())])

    result = _generate(_client())

    assert len(requests) == 2
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in requests[1]
    assert requests[1] == {
        key: value for key, value in requests[0].items() if key != "response_format"
    }
    system, user = requests[1]["messages"]
    assert system["role"] == "system"
    assert "Select the matching tiles." in system["content"]
    assert "JSON" in system["content"]
    assert json.dumps(ImageAreaSelectChallenge.model_json_schema(), ensure_ascii=False) in (
        system["content"]
    )
    assert user == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Find the matching tile in this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
        ],
    }
    assert isinstance(result.parsed, ImageAreaSelectChallenge)
    assert [point.model_dump() for point in result.parsed.points] == [{"x": 20, "y": 40}]
    assert result.text == _success()["choices"][0]["message"]["content"]


def test_json_mode_capability_cache_is_limited_to_the_same_client_and_model(monkeypatch):
    requests = _mock_endpoint(
        monkeypatch,
        [(400, {"code": 20024})] + [(200, _success())] * 4,
    )
    client = _client()

    _generate(client, "model-a")
    _generate(client, "model-a")
    _generate(client, "model-b")
    _generate(_client(), "model-a")

    assert len(requests) == 5
    assert [request["model"] for request in requests] == [
        "model-a",
        "model-a",
        "model-a",
        "model-b",
        "model-a",
    ]
    assert ["response_format" in request for request in requests] == [
        True,
        False,
        False,
        True,
        True,
    ]


def test_supported_json_mode_needs_only_one_request(monkeypatch):
    requests = _mock_endpoint(monkeypatch, [(200, _success())])

    result = _generate(_client())

    assert len(requests) == 1
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert result.parsed.points[0].model_dump() == {"x": 20, "y": 40}


@pytest.mark.parametrize(
    "status,error",
    [
        (400, {"code": 20015, "message": '"messages" are illegal: 151652 is not in list'}),
        (400, {"error": {"code": "20015", "message": "Invalid image messages"}}),
        (401, {"error": {"code": "invalid_api_key"}}),
        (403, {"code": "permission_denied"}),
        (404, {"message": "Model not found"}),
        (413, {"message": "Payload too large"}),
        (422, {"message": "Invalid request"}),
    ],
)
def test_permanent_request_errors_abort_without_fallback(monkeypatch, status, error):
    requests = _mock_endpoint(monkeypatch, [(status, error)])

    with pytest.raises(LLMRequestAbort, match=rf"HTTP {status}"):
        _generate(_client())

    assert len(requests) == 1
    assert requests[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("second_code", [20024, 20015])
def test_fallback_rejection_aborts_instead_of_looping(monkeypatch, second_code):
    requests = _mock_endpoint(
        monkeypatch,
        [(400, {"code": 20024}), (400, {"code": second_code})],
    )

    with pytest.raises(LLMRequestAbort, match="HTTP 400"):
        _generate(_client())

    assert len(requests) == 2
    assert "response_format" not in requests[1]


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
def test_transient_errors_preserve_http_status_error(monkeypatch, status):
    requests = _mock_endpoint(monkeypatch, [(status, {"code": 20024})])

    with pytest.raises(httpx.HTTPStatusError) as raised:
        _generate(_client())

    assert raised.value.response.status_code == status
    assert len(requests) == 1


@pytest.mark.parametrize("during_fallback", [False, True])
def test_timeout_keeps_budget_and_original_exception(monkeypatch, during_fallback):
    timeout = httpx.ReadTimeout("read timed out")
    replies = [(400, {"code": 20024}), timeout] if during_fallback else [timeout]
    requests = _mock_endpoint(monkeypatch, replies)

    with pytest.raises(TimeoutError, match=r"50 seconds \(ReadTimeout\)") as raised:
        _generate(_client())

    assert raised.value.__cause__ is timeout
    assert len(requests) == (2 if during_fallback else 1)
