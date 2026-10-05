"""Offline regressions for the minimal model, grounding and store-session fixes."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import cv2
import httpx
import numpy as np
import pytest
from hcaptcha_challenger.models import ChallengeSignal, ImageAreaSelectChallenge, PointCoordinate

from extensions import hcaptcha_adapter as captcha
from extensions.llm_adapter import _GLMAsyncModels
from extensions.llm_errors import (
    LLMConfigurationError,
    LLMRequestAbort,
    LLMResponseError,
    llm_request_boundary,
)
from extensions.spatial_coordinates import to_page_point, unique_points


BBOX = {"x": 1200, "y": 800, "width": 500, "height": 400}
ANSWER = {
    "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":[[10,20,30,60]]}'}}]
}


def _model(monkeypatch, replies):
    requests, real_client = [], httpx.AsyncClient

    def respond(request):
        requests.append(json.loads(request.content))
        status, body = replies[len(requests) - 1]
        return httpx.Response(status, json=body)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw)
    )
    client = _GLMAsyncModels(
        SimpleNamespace(
            GLM_BASE_URL="https://example.invalid/v1",
            GLM_REQUEST_TIMEOUT_SECONDS=50,
            GLM_API_KEY=SimpleNamespace(get_secret_value=lambda: "offline-key"),
            GLM_MAX_TOKENS=2048,
        ),
        {},
    )
    config = SimpleNamespace(
        response_schema=ImageAreaSelectChallenge,
        system_instruction=None,
        temperature=None,
        thinking_config=None,
    )
    contents = SimpleNamespace(
        parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b"image", mime_type="image/png"))]
    )
    return lambda: client.generate_content("vision-model", contents, config=config), requests


def test_json_fallback_preserves_images_schema_and_remembers_endpoint_capability(monkeypatch):
    generate, requests = _model(monkeypatch, [(400, {"code": 20024}), (200, ANSWER), (200, ANSWER)])

    async def invoke():
        first = await generate()
        await generate()
        return first

    result = asyncio.run(invoke())
    assert ["response_format" in request for request in requests] == [True, False, False]
    assert requests[0]["messages"] == requests[1]["messages"]
    assert requests[1]["messages"][1]["content"][0]["type"] == "image_url"
    assert "JSON Schema" in requests[1]["messages"][0]["content"]
    assert requests[1]["max_tokens"] == 2048
    assert result.parsed.points[0].model_dump() == {"x": 20, "y": 40}


@pytest.mark.parametrize(
    "replies,error",
    [
        ([(400, {"code": 20015})], LLMRequestAbort),
        ([(401, {"error": {"code": "invalid_key"}})], LLMRequestAbort),
        ([(400, {"code": 20024}), (400, {"code": 20024})], LLMRequestAbort),
        (
            [(200, {"choices": [{**ANSWER["choices"][0], "finish_reason": "length"}]})],
            LLMResponseError,
        ),
        (
            [(200, {"choices": [{"message": {"refusal": "refused", "content": "{}"}}]})],
            LLMResponseError,
        ),
        ([(200, {"choices": [{"message": {"content": ""}}]})], LLMResponseError),
        (
            [(200, {"choices": [{"message": {"content": '{"points":"not-a-point-list"}'}}]})],
            LLMResponseError,
        ),
    ],
)
def test_rejected_requests_do_not_loop_or_accept_truncated_json(monkeypatch, replies, error):
    generate, requests = _model(monkeypatch, replies)
    with pytest.raises(error):
        asyncio.run(generate())
    assert len(requests) == len(replies)


@pytest.mark.parametrize(
    "signal,expected",
    [
        (LLMRequestAbort("sample rejected"), RuntimeError),
        (
            LLMRequestAbort("configuration rejected", is_configuration_error=True),
            LLMConfigurationError,
        ),
        (asyncio.CancelledError(), asyncio.CancelledError),
    ],
)
def test_llm_boundary_cleans_up_before_conversion_and_preserves_cancellation(signal, expected):
    events = []

    @asynccontextmanager
    async def inner_context():
        try:
            yield
        finally:
            events.append("cleanup")

    async def invoke():
        try:
            async with llm_request_boundary(), inner_context():
                raise signal
        except BaseException as error:
            events.append(type(error))
            raise

    with pytest.raises(expected) as raised:
        asyncio.run(invoke())
    assert events == ["cleanup", expected]
    if isinstance(signal, asyncio.CancelledError):
        assert raised.value is signal


def test_normalized_points_map_once_deduplicate_and_reject_css_edges():
    point = PointCoordinate(x=500, y=500)
    mapped = to_page_point(point, BBOX)
    assert (mapped.x, mapped.y) == (1450, 1000)
    assert (point.x, point.y) == (500, 500)
    assert unique_points([mapped, mapped]) == [mapped]
    edge = to_page_point(PointCoordinate(x=1000, y=500), BBOX)
    assert captcha._point_answer_validation_error(
        [edge], challenge_bbox=BBOX, clickable_bounds=None
    )


@pytest.mark.parametrize("value", [-1, 1001, float("nan")])
def test_invalid_normalized_coordinates_are_not_clamped(value):
    with pytest.raises(ValueError):
        to_page_point(SimpleNamespace(x=value, y=500), BBOX)


@pytest.mark.parametrize("mirrored", [False, True])
def test_source_regions_are_excluded_without_discarding_the_opposite_side(
    tmp_path, monkeypatch, mirrored
):
    image = np.full((400, 500, 3), (60, 30, 10), dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (80, 100), (240, 240, 240), 4)
    cv2.rectangle(image, (360, 200), (410, 250), (240, 240, 240), 4)
    source_x, target_x = (444, 114) if mirrored else (55, 385)
    if mirrored:
        image = cv2.flip(image, 1)
    path = tmp_path / "outline.png"
    assert cv2.imwrite(str(path), image)
    monkeypatch.setattr(captcha, "_detect_task_canvas_origin", lambda _path: (0, 0))
    entities = [SimpleNamespace(coords=[source_x, 70], size=[70, 80])]
    targets = captcha._extract_outline_targets(path, source_entities=entities)
    assert len(targets) == 1 and targets[0][1] == pytest.approx((target_x, 225), abs=1)
    entities[0].coords = ["invalid", 70]
    assert captcha._extract_outline_targets(path, source_entities=entities) == []
    assert (
        captcha._entity_centers(SimpleNamespace(tasklist=[SimpleNamespace(entities=entities)]), 0)
        == []
    )


def test_answer_is_not_clicked_after_challenge_bounds_change(tmp_path, monkeypatch):
    class Arm:
        async def challenge_image_label_select(self, _job):
            raise AssertionError("Point patch was not installed")

        async def challenge_image_drag_drop(self, _job):
            raise AssertionError("Unexpected drag")

    monkeypatch.setattr(captcha, "RoboticArm", Arm)
    monkeypatch.setattr(captcha, "_apply_empty_checkcaptcha_patch", lambda: None)
    monkeypatch.setattr(captcha, "_detect_task_canvas_bounds", lambda _path: None)
    answer = ImageAreaSelectChallenge(
        challenge_prompt="Select circles", points=[PointCoordinate(x=500, y=500)]
    )
    monkeypatch.setattr(captcha, "_request_spatial_response", AsyncMock(return_value=answer))
    view = SimpleNamespace(bounding_box=AsyncMock(side_effect=[BBOX, BBOX, {**BBOX, "x": 1220}]))
    page = SimpleNamespace(wait_for_timeout=AsyncMock(), mouse=SimpleNamespace(click=AsyncMock()))
    arm = SimpleNamespace(
        page=page,
        captcha_payload=object(),
        check_crumb_count=AsyncMock(return_value=1),
        get_challenge_frame_locator=AsyncMock(
            return_value=SimpleNamespace(locator=lambda _s: view)
        ),
        config=SimpleNamespace(
            create_cache_key=lambda _p: tmp_path,
            WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS=0,
            SPATIAL_POINT_REASONER_MODEL="Qwen/Qwen3-VL-32B-Instruct",
        ),
        _capture_spatial_mapping=AsyncMock(
            return_value=(tmp_path / "raw.png", tmp_path / "grid.png")
        ),
        _match_user_prompt=lambda _job: "Select circles",
        _spatial_point_reasoner=SimpleNamespace(cache_response=Mock()),
    )
    captcha.apply_hcaptcha_drag_patch()
    with pytest.raises(ValueError, match="bounds changed"):
        asyncio.run(Arm.challenge_image_label_select(arm, "select"))
    page.mouse.click.assert_not_awaited()


@pytest.fixture
def auth_module(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-key")
    from services import epic_authorization_service

    return epic_authorization_service


def _store_page():
    link = SimpleNamespace(
        count=AsyncMock(return_value=1),
        is_visible=AsyncMock(return_value=True),
        get_attribute=AsyncMock(
            side_effect=lambda name, **_kw: "/login?state=offline" if name == "href" else None
        ),
        click=AsyncMock(),
    )
    return (
        SimpleNamespace(
            url="https://store.epicgames.com/free-games", get_by_role=Mock(return_value=link)
        ),
        link,
    )


@pytest.mark.parametrize("authenticates", [False, True])
@pytest.mark.parametrize("entry", ["auth", "claims"])
def test_official_handoff_must_be_followed_by_true_store_marker(
    auth_module, monkeypatch, authenticates, entry
):
    page, link = _store_page()
    clock, marker = [0.0], ["false"]

    async def wait(ms):
        clock[0] += ms / 1000

    async def click(**_kwargs):
        marker[0] = "true" if authenticates else "false"

    page.wait_for_timeout, link.click.side_effect = wait, click
    monkeypatch.setattr(auth_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    if entry == "claims":
        from services import epic_games_service

        monkeypatch.setattr(epic_games_service, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        auth = epic_games_service.EpicAgent(page)
        wait_for_session = auth._wait_for_claim_page_login_state
    else:
        auth = auth_module.EpicAuthorization(page)
        wait_for_session = auth._ensure_store_session_ready
    auth._get_login_status = AsyncMock(side_effect=lambda **_kw: marker[0])
    auth._has_account_session = AsyncMock(return_value=True)
    if authenticates:
        result = asyncio.run(wait_for_session())
        assert result == ("true" if entry == "claims" else None)
        assert clock[0] == 8
    else:
        if entry == "claims":
            assert asyncio.run(wait_for_session()) == "false"
        else:
            with pytest.raises(RuntimeError, match="did not confirm"):
                asyncio.run(wait_for_session())
        assert clock[0] == 45
    link.click.assert_awaited_once_with(timeout=5000, no_wait_after=True)
    page.get_by_role.assert_called_once_with("link", name="Sign in", exact=True)
    auth._has_account_session.assert_not_awaited()


@pytest.mark.parametrize("kind", ["external", "hidden", "ambiguous"])
def test_sign_in_requires_one_visible_official_link(auth_module, kind):
    page, link = _store_page()
    if kind == "external":
        link.get_attribute.side_effect = lambda name, **_kw: (
            "https://evil.invalid/login" if name == "href" else None
        )
    elif kind == "hidden":
        link.is_visible.return_value = False
    else:
        link.count.return_value = 2
    assert asyncio.run(auth_module.start_store_sign_in(page)) is False
    link.click.assert_not_awaited()


def test_email_captcha_rejection_preempts_another_solve(auth_module, monkeypatch):
    auth = auth_module.EpicAuthorization(SimpleNamespace(url="https://www.epicgames.com/id/login"))
    solver = AsyncMock()
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solver)
    response = SimpleNamespace(
        url="https://www.epicgames.com/id/api/email",
        status=400,
        request=SimpleNamespace(method="POST"),
        json=AsyncMock(
            return_value={"errorCode": "errors.com.epicgames.accountportal.captcha_invalid"}
        ),
    )
    asyncio.run(auth._on_response_anything(response))
    with pytest.raises(RuntimeError, match="captcha_invalid"):
        asyncio.run(auth._await_login_outcome(auth.page.url, object()))
    solver.assert_not_awaited()


def test_login_captcha_budget_is_shared_and_exceptions_count(auth_module, monkeypatch):
    auth = auth_module.EpicAuthorization(SimpleNamespace())
    solver = AsyncMock(
        side_effect=[ChallengeSignal.FAILURE, TimeoutError(), ChallengeSignal.SUCCESS]
    )
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solver)

    async def invoke():
        await auth._wait_for_login_challenge(object(), context="login:1", timeout_seconds=10)
        with pytest.raises(TimeoutError):
            await auth._wait_for_login_challenge(object(), context="login_mfa", timeout_seconds=10)
        await auth._wait_for_login_challenge(object(), context="login_mfa", timeout_seconds=10)
        with pytest.raises(auth_module.EpicCaptchaBudgetExceededError):
            await auth._wait_for_login_challenge(object(), context="login:2", timeout_seconds=10)

    asyncio.run(invoke())
    assert solver.await_count == 3
