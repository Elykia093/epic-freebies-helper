import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from extensions import hcaptcha_adapter, llm_adapter
from extensions.llm_adapter import _GLMAsyncModels, _response_diagnostics
from extensions.llm_errors import LLMResponseError


class Selection(BaseModel):
    selection: Literal["chosen"]


def _client(**overrides):
    settings = {
        "GLM_BASE_URL": "https://example.invalid/v1",
        "GLM_API_KEY": SimpleNamespace(get_secret_value=lambda: "private-api-key"),
        "GLM_REQUEST_TIMEOUT_SECONDS": 50,
    }
    settings.update(overrides)
    return _GLMAsyncModels(settings=SimpleNamespace(**settings), storage={})


def _config():
    return SimpleNamespace(
        response_schema=Selection,
        system_instruction=None,
        temperature=None,
        thinking_config=None,
    )


def _response(content='{"selection":"chosen"}', **choice_overrides):
    choice = {"message": {"content": content}, "finish_reason": "stop"}
    choice.update(choice_overrides)
    return {"choices": [choice]}


def _mock_response(monkeypatch, body):
    requests = []
    real_async_client = httpx.AsyncClient

    def handle(request):
        requests.append(json.loads(request.content))
        assert len(requests) == 1, "Response validation must not issue another HTTP request"
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    return requests


def _generate(client):
    return asyncio.run(client.generate_content("vision-model", [], config=_config()))


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_explicit_budget_settings_are_forwarded(monkeypatch, enable_thinking):
    requests = _mock_response(monkeypatch, _response())

    _generate(
        _client(
            GLM_ENABLE_THINKING=enable_thinking,
            GLM_THINKING_BUDGET=2048,
            GLM_MAX_TOKENS=4096,
        )
    )

    assert requests[0]["enable_thinking"] is enable_thinking
    assert requests[0]["thinking_budget"] == 2048
    assert requests[0]["max_tokens"] == 4096


@pytest.mark.parametrize("explicit_none", [False, True])
def test_unset_budget_settings_do_not_change_request_parameters(monkeypatch, explicit_none):
    requests = _mock_response(monkeypatch, _response())
    overrides = (
        {"GLM_ENABLE_THINKING": None, "GLM_THINKING_BUDGET": None, "GLM_MAX_TOKENS": None}
        if explicit_none
        else {}
    )

    _generate(_client(**overrides))

    assert not {"enable_thinking", "thinking_budget", "max_tokens"} & requests[0].keys()


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_truncated_response_is_rejected_even_when_json_is_complete(monkeypatch, finish_reason):
    requests = _mock_response(monkeypatch, _response(finish_reason=finish_reason))

    with pytest.raises(LLMResponseError, match="truncated"):
        _generate(_client())

    assert len(requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        _response(message={"reasoning_content": '{"selection":"chosen"}'}),
        _response(""),
        _response(" \n\t "),
        _response("{}"),
        _response("ordinary prose with no structured answer"),
        {"choices": []},
        {},
    ],
    ids=[
        "reasoning-only",
        "empty",
        "whitespace",
        "empty-object",
        "prose",
        "no-choices",
        "empty-body",
    ],
)
def test_response_without_a_usable_structured_answer_is_rejected(monkeypatch, body):
    _mock_response(monkeypatch, body)

    with pytest.raises(LLMResponseError):
        _generate(_client())


@pytest.mark.parametrize("finish_reason", ["content_filter", "safety"])
def test_filtered_response_is_rejected_even_when_it_contains_an_answer(monkeypatch, finish_reason):
    _mock_response(monkeypatch, _response(finish_reason=finish_reason))

    with pytest.raises(LLMResponseError, match="refused or filtered"):
        _generate(_client())


def test_explicit_refusal_is_rejected(monkeypatch):
    body = _response()
    body["choices"][0]["message"]["refusal"] = "private-refusal-text"
    _mock_response(monkeypatch, body)

    with pytest.raises(LLMResponseError, match="refused or filtered") as raised:
        _generate(_client())

    assert "private-refusal-text" not in str(raised.value)


@pytest.mark.parametrize("include_finish_reason", [False, True])
def test_valid_response_accepts_missing_finish_reason(monkeypatch, include_finish_reason):
    body = _response()
    if not include_finish_reason:
        del body["choices"][0]["finish_reason"]
    body["choices"][0]["message"]["reasoning_content"] = '{"selection":"wrong"}'
    _mock_response(monkeypatch, body)

    result = _generate(_client())

    assert isinstance(result.parsed, Selection)
    assert result.parsed.selection == "chosen"


@pytest.mark.parametrize(
    "invalid_content", ["private-answer-text", '{"selection":"private-answer-text"}']
)
def test_response_diagnostics_and_validation_errors_do_not_log_sensitive_text(
    monkeypatch, invalid_content
):
    messages = []

    def record(message, *args, **_kwargs):
        messages.append(message.format(*args))

    monkeypatch.setattr(
        llm_adapter, "logger", SimpleNamespace(info=record, warning=record, error=record)
    )
    body = _response(invalid_content)
    body["choices"][0]["message"]["reasoning_content"] = "private-reasoning-text"
    body["api_key"] = "private-api-key"
    _mock_response(monkeypatch, body)

    with pytest.raises(LLMResponseError) as raised:
        _generate(_client())

    diagnostic_text = "\n".join(messages) + str(raised.value)
    assert "LLM response" in diagnostic_text
    assert "content_chars" in diagnostic_text
    assert "reasoning_chars" in diagnostic_text
    for secret in ("private-answer-text", "private-reasoning-text", "private-api-key"):
        assert secret not in diagnostic_text
    if invalid_content.startswith("{"):
        assert "schema validation" in str(raised.value)
        assert raised.value.__suppress_context__


def test_diagnostics_keep_only_safe_metadata_and_preserve_zero_usage():
    body = _response("private-answer", finish_reason="private-finish-reason")
    body["choices"][0]["message"]["reasoning_content"] = "private-reasoning"
    body["choices"][0]["message"]["refusal"] = "private-refusal"
    body["api_key"] = "private-api-key"
    body["usage"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "private-extra": "private-usage-text",
    }

    diagnostics = _response_diagnostics(body)

    assert diagnostics == {
        "finish_reason": "unknown",
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "content_chars": len("private-answer"),
        "reasoning_chars": len("private-reasoning"),
        "has_refusal": True,
    }
    assert "private-" not in json.dumps(diagnostics)


@pytest.mark.parametrize("invalid_count", [True, False, -1, 1.5, "12", None])
def test_diagnostics_reject_non_integer_and_negative_usage(invalid_count):
    body = _response()
    body["usage"] = {
        "prompt_tokens": invalid_count,
        "completion_tokens": invalid_count,
        "completion_tokens_details": {"reasoning_tokens": invalid_count},
    }

    diagnostics = _response_diagnostics(body)

    assert diagnostics["input_tokens"] is None
    assert diagnostics["output_tokens"] is None
    assert diagnostics["reasoning_tokens"] is None


def test_diagnostics_support_text_parts_and_flat_reasoning_usage():
    body = _response([{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}])
    body["usage"] = {"reasoning_tokens": 7}

    diagnostics = _response_diagnostics(body)

    assert diagnostics["content_chars"] == 3
    assert diagnostics["reasoning_tokens"] == 7


@pytest.fixture
def settings_factory(monkeypatch, tmp_path):
    # Isolate import-time setup without replacing EpicSettings or any of its validators.
    monkeypatch.setattr(os, "environ", {"GEMINI_API_KEY": "fixture-key", "LLM_PROVIDER": "gemini"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(llm_adapter, "apply_llm_patch", lambda _settings: None)
    monkeypatch.setattr(hcaptcha_adapter, "apply_hcaptcha_drag_patch", lambda: None)
    path = Path(__file__).resolve().parents[1] / "app" / "settings.py"
    spec = importlib.util.spec_from_file_location("llm_budget_settings_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def build(**overrides):
        values = {
            "_env_file": None,
            "LLM_PROVIDER": "glm",
            "GLM_API_KEY": "fixture-key",
            "GEMINI_API_KEY": "fixture-key",
            "GLM_REQUEST_TIMEOUT_SECONDS": 50,
            "EXECUTION_TIMEOUT": 120,
            "RESPONSE_TIMEOUT": 30,
        }
        values.update(overrides)
        return module.EpicSettings(**values)

    return build


@pytest.mark.parametrize(
    "http_timeout,execution_timeout", [(50, 120), (90, 240), (50, 103), (120, 243)]
)
def test_glm_timeouts_accept_sufficient_two_attempt_budget(
    settings_factory, http_timeout, execution_timeout
):
    settings = settings_factory(
        GLM_REQUEST_TIMEOUT_SECONDS=http_timeout, EXECUTION_TIMEOUT=execution_timeout
    )

    assert settings.GLM_REQUEST_TIMEOUT_SECONDS == http_timeout
    assert settings.EXECUTION_TIMEOUT == execution_timeout


@pytest.mark.parametrize("http_timeout,execution_timeout", [(120, 240), (50, 102)])
def test_glm_timeouts_reject_insufficient_two_attempt_budget(
    settings_factory, http_timeout, execution_timeout
):
    with pytest.raises(ValidationError, match="EXECUTION_TIMEOUT must be at least"):
        settings_factory(
            GLM_REQUEST_TIMEOUT_SECONDS=http_timeout, EXECUTION_TIMEOUT=execution_timeout
        )


@pytest.mark.parametrize(
    "field", ["GLM_REQUEST_TIMEOUT_SECONDS", "EXECUTION_TIMEOUT", "RESPONSE_TIMEOUT"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_timeouts_reject_nonfinite_values(settings_factory, field, value):
    with pytest.raises(ValidationError) as raised:
        settings_factory(**{field: value})

    assert any(error["loc"] == (field,) for error in raised.value.errors())


def test_gemini_does_not_use_the_glm_request_retry_budget(settings_factory):
    settings = settings_factory(
        LLM_PROVIDER="gemini", GLM_REQUEST_TIMEOUT_SECONDS=120, EXECUTION_TIMEOUT=30
    )

    assert settings.LLM_PROVIDER == "gemini"
    assert settings.EXECUTION_TIMEOUT == 30


def test_settings_preserve_optional_budget_defaults_and_explicit_false(settings_factory):
    defaults = settings_factory()
    explicit = settings_factory(
        GLM_ENABLE_THINKING=False, GLM_THINKING_BUDGET=128, GLM_MAX_TOKENS=1
    )

    assert defaults.GLM_ENABLE_THINKING is None
    assert defaults.GLM_THINKING_BUDGET is None
    assert defaults.GLM_MAX_TOKENS is None
    assert explicit.GLM_ENABLE_THINKING is False
    assert explicit.GLM_THINKING_BUDGET == 128
    assert explicit.GLM_MAX_TOKENS == 1
