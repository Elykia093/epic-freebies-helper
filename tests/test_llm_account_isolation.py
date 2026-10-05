"""Exercise real LLM failures through the browser boundary and account dispatcher."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from loguru import logger

from accounts import mask_email
from extensions.llm_adapter import _GLMAsyncModels
from extensions.llm_errors import LLMConfigurationError, llm_request_boundary


ACCOUNTS = [("first@example.test", "unused"), ("second@example.test", "unused")]


def _batch_scenario(monkeypatch, first_status):
    current, visited, requests = [None], [], []
    real_client = httpx.AsyncClient

    def respond(request):
        payload = json.loads(request.content)
        status = first_status if current[0] == ACCOUNTS[0][0] else 200
        requests.append((status, payload["model"], request.headers["Authorization"]))
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "Request rejected"}})
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]},
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw)
    )
    model = _GLMAsyncModels(
        SimpleNamespace(
            GLM_BASE_URL="https://example.invalid/v1",
            GLM_API_KEY=SimpleNamespace(get_secret_value=lambda: "offline-key"),
            GLM_REQUEST_TIMEOUT_SECONDS=50,
            GLM_MAX_TOKENS=None,
        ),
        {},
    )
    config = SimpleNamespace(
        response_schema=None, system_instruction=None, temperature=None, thinking_config=None
    )

    async def account_request(**_kwargs):
        visited.append(current[0])
        contents = SimpleNamespace(
            parts=[
                SimpleNamespace(
                    inline_data=SimpleNamespace(data=current[0].encode(), mime_type="image/png")
                )
            ]
        )
        async with llm_request_boundary():
            await model.generate_content("same-vision-model", contents, config=config)

    # Keep the production dispatcher intact without importing deploy's file-logging setup.
    path = Path(__file__).resolve().parents[1] / "app" / "deploy.py"
    function = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_multiple_accounts"
    )
    namespace = {
        "logger": logger,
        "mask_email": mask_email,
        "swap_account": lambda email, password: current.__setitem__(0, email),
        "execute_browser_tasks_with_notification": account_request,
        "LLMConfigurationError": LLMConfigurationError,
        "RATE_LIMITED_OUTCOME": "rate_limited",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return lambda: asyncio.run(namespace["execute_multiple_accounts"](ACCOUNTS)), visited, requests


@pytest.mark.parametrize("status", [400, 403, 413, 415, 422])
def test_image_rejection_stops_one_account_but_runs_the_next(monkeypatch, status):
    run, visited, requests = _batch_scenario(monkeypatch, status)

    with pytest.raises(RuntimeError) as caught:
        run()

    assert visited == [email for email, _password in ACCOUNTS]
    assert [code for code, _model, _auth in requests] == [status, 200]
    assert len({(model, auth) for _status, model, auth in requests}) == 1
    assert not isinstance(caught.value, LLMConfigurationError)
    assert "1 of 2 account(s) failed" in str(caught.value)


@pytest.mark.parametrize("status", [401, 404, 405])
def test_shared_configuration_rejection_stops_remaining_accounts(monkeypatch, status):
    run, visited, requests = _batch_scenario(monkeypatch, status)

    with pytest.raises(LLMConfigurationError):
        run()

    assert visited == [ACCOUNTS[0][0]]
    assert [code for code, _model, _auth in requests] == [status]
