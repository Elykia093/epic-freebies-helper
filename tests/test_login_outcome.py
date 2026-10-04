import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hcaptcha_challenger.models import ChallengeSignal


LOGIN_URL = "https://www.epicgames.com/id/login"
REJECTION = "Incorrect response. Please refresh the page."


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.on_wait = lambda: None

    async def wait(self, milliseconds):
        self.now += milliseconds / 1000
        self.on_wait()


class FakeLocator:
    def __init__(self, visible=False, text=""):
        self.visible = visible
        self.text = text

    @property
    def first(self):
        return self

    async def is_visible(self, **_kwargs):
        return self.visible() if callable(self.visible) else self.visible

    async def count(self):
        return int(await self.is_visible())

    async def evaluate(self, _expression):
        return await self.is_visible()

    async def inner_text(self, **_kwargs):
        return self.text


class FakeFrame:
    def __init__(self, url, *, active=True, visible=True):
        self.url = url
        self.active = active
        self.visible = visible

    def locator(self, selector):
        return FakeLocator(self.active if "challenge-view" in selector else False)

    async def frame_element(self):
        return FakeLocator(self.visible)


class FakePage:
    def __init__(self, clock, *, body="", url=LOGIN_URL, frames=None):
        self.url = url
        self.frames = frames or []
        self.body = body
        self.wait_for_timeout = clock.wait

    def locator(self, selector):
        return FakeLocator(True, self.body) if selector == "body" else FakeLocator()


@pytest.fixture
def auth_module(monkeypatch, tmp_path):
    # Isolate startup settings and the external TOTP boundary, not the login state machine.
    settings_module = ModuleType("settings")
    settings_module.SCREENSHOTS_DIR = tmp_path
    settings_module.settings = SimpleNamespace(EXECUTION_TIMEOUT=120, RESPONSE_TIMEOUT=30)
    totp_module = ModuleType("services.epic_totp_service")
    totp_module.redact_totp_inputs = AsyncMock()
    totp_module.submit_totp_challenge = AsyncMock(return_value=True)
    totp_module.totp_login_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "settings", settings_module)
    monkeypatch.setitem(sys.modules, "services.epic_totp_service", totp_module)
    path = Path(__file__).resolve().parents[1] / "app/services/epic_authorization_service.py"
    spec = importlib.util.spec_from_file_location("login_outcome_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    messages = []

    def record(message, *args, **_kwargs):
        messages.append(message.format(*args))

    monkeypatch.setattr(
        module,
        "logger",
        SimpleNamespace(info=record, debug=record, warning=record, error=record, success=record),
    )
    module.test_messages = messages
    return module


@pytest.fixture
def login_state(auth_module, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(auth_module, "time", SimpleNamespace(monotonic=lambda: clock.now))

    def build(**page_kwargs):
        page = FakePage(clock, **page_kwargs)
        authorization = auth_module.EpicAuthorization(page)
        authorization._get_login_status = AsyncMock(return_value="false")
        return authorization, page, clock

    return build


def _active_frame(**kwargs):
    return FakeFrame(
        "https://newassets.hcaptcha.com/captcha/hcaptcha.html?frame=challenge", **kwargs
    )


def test_initial_captcha_success_observes_residual_frame_without_resolving_again(
    auth_module, login_state, monkeypatch
):
    auth, _page, clock = login_state(frames=[_active_frame()])
    solver = AsyncMock(return_value=ChallengeSignal.SUCCESS)
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solver)

    def successful_login():
        if clock.now >= 101.5:
            auth._is_login_success_signal.put_nowait({"accountId": "offline-account"})

    clock.on_wait = successful_login
    asyncio.run(auth._await_login_outcome(LOGIN_URL, object(), challenge_succeeded=True))

    solver.assert_not_awaited()
    assert clock.now == 101.5


def _prepare_login(auth_module, auth, page, monkeypatch, signal):
    auth_module.settings.EPIC_EMAIL = "offline@example.invalid"
    auth_module.settings.EPIC_PASSWORD = SimpleNamespace(
        get_secret_value=lambda: "offline-password"
    )
    monkeypatch.setattr(auth_module, "AgentV", lambda **_kwargs: object())
    monkeypatch.setattr(
        auth_module, "expect", lambda _locator: SimpleNamespace(to_be_visible=AsyncMock())
    )
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", AsyncMock(return_value=signal))
    page.goto = AsyncMock()
    page.click = AsyncMock()
    original_locator = page.locator
    page.locator = lambda selector: (
        SimpleNamespace(fill=AsyncMock())
        if selector in {"#email", "#password"}
        else original_locator(selector)
    )
    for name in (
        "_wait_for_login_form",
        "_submit_login_or_accept_challenge",
        "_handle_right_account_validation",
        "_goto_claim_page",
        "_ensure_store_session_ready",
        "_await_login_outcome",
    ):
        monkeypatch.setattr(auth, name, AsyncMock())


@pytest.mark.parametrize("signal", [ChallengeSignal.SUCCESS, ChallengeSignal.FAILURE])
def test_login_passes_the_initial_captcha_result_to_outcome_observation(
    auth_module, login_state, monkeypatch, signal
):
    auth, page, _clock = login_state()
    _prepare_login(auth_module, auth, page, monkeypatch, signal)

    assert asyncio.run(auth._login()) is True

    auth._await_login_outcome.assert_awaited_once()
    assert auth._await_login_outcome.await_args.kwargs["challenge_succeeded"] is (
        signal is ChallengeSignal.SUCCESS
    )


def test_outer_login_timeout_does_not_resolve_a_remaining_checkbox(
    auth_module, login_state, monkeypatch
):
    auth, page, _clock = login_state(
        body="One more step. I am human.",
        frames=[
            FakeFrame(
                "https://newassets.hcaptcha.com/captcha/hcaptcha.html?frame=checkbox", active=False
            )
        ],
    )
    _prepare_login(auth_module, auth, page, monkeypatch, ChallengeSignal.SUCCESS)
    auth._await_login_outcome.side_effect = auth_module.PlaywrightTimeoutError("login timed out")
    monkeypatch.setattr(auth, "_resubmit_password_form", AsyncMock(return_value=True))
    monkeypatch.setattr(auth_module.time, "time", lambda: 1000, raising=False)
    page.screenshot = AsyncMock()
    assert asyncio.run(auth._has_visible_hcaptcha()) is True
    assert asyncio.run(auth._has_active_hcaptcha_challenge()) is False

    assert asyncio.run(auth._login()) is None

    auth_module.wait_for_challenge_signal.assert_awaited_once()
    assert auth._await_login_outcome.await_count == 2
    auth._resubmit_password_form.assert_awaited_once()


def test_success_from_internal_solver_gets_the_same_observation_window(
    auth_module, login_state, monkeypatch
):
    auth, _page, clock = login_state(frames=[_active_frame()])
    solver = AsyncMock(return_value=ChallengeSignal.SUCCESS)
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solver)
    clock.on_wait = lambda: (
        auth._is_login_success_signal.put_nowait({"accountId": "offline-account"})
        if clock.now >= 101.5
        else None
    )

    asyncio.run(auth._await_login_outcome(LOGIN_URL, object()))

    solver.assert_awaited_once()
    assert clock.now == 101.5


def test_a_new_active_challenge_can_be_solved_after_three_seconds(
    auth_module, login_state, monkeypatch
):
    auth, page, clock = login_state()
    page.frames = [_active_frame(active=lambda: clock.now >= 103.0)]
    solve_times = []

    async def solve(_agent, **_kwargs):
        solve_times.append(clock.now)
        auth._is_login_success_signal.put_nowait({"accountId": "offline-account"})
        return ChallengeSignal.SUCCESS

    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solve)
    asyncio.run(auth._await_login_outcome(LOGIN_URL, object(), challenge_succeeded=True))

    assert solve_times == [103.0]


@pytest.mark.parametrize("challenge_succeeded", [False, True])
def test_explicit_login_rejection_fails_before_solver_even_during_observation(
    auth_module, login_state, monkeypatch, challenge_succeeded
):
    auth, _page, clock = login_state(body=REJECTION, frames=[_active_frame()])
    solver = AsyncMock(return_value=ChallengeSignal.SUCCESS)
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solver)

    with pytest.raises(RuntimeError, match="Epic rejected the login verification response"):
        asyncio.run(
            auth._await_login_outcome(LOGIN_URL, object(), challenge_succeeded=challenge_succeeded)
        )

    solver.assert_not_awaited()
    assert clock.now < 101.0


@pytest.mark.parametrize(
    "url,active,expected",
    [
        ("https://newassets.hcaptcha.com/captcha/hcaptcha.html?frame=checkbox", False, False),
        ("https://newassets.hcaptcha.com/captcha/hcaptcha.html?frame=challenge", True, True),
        ("https://newassets.hcaptcha.com/captcha/hcaptcha.html#frame=challenge", True, True),
        ("https://newassets.hcaptcha.com/captcha/hcaptcha.html?frame=challenge", False, False),
        ("https://hcaptcha.com.evil.invalid/hcaptcha.html?frame=challenge", True, False),
    ],
)
def test_only_a_real_visible_challenge_view_is_active(login_state, url, active, expected):
    auth, _page, _clock = login_state(
        body="One more step. Please complete a security check. I am human.",
        frames=[FakeFrame(url, active=active)],
    )

    assert asyncio.run(auth._has_active_hcaptcha_challenge()) is expected


def test_security_prompt_without_a_challenge_view_is_not_active(login_state):
    auth, _page, _clock = login_state(body="Verify you are human. I am human.")

    assert asyncio.run(auth._has_active_hcaptcha_challenge()) is False


def test_hidden_iframe_with_visible_inner_challenge_is_not_active(login_state):
    auth, _page, _clock = login_state(frames=[_active_frame(active=True, visible=False)])

    assert asyncio.run(auth._has_active_hcaptcha_challenge()) is False


@pytest.mark.parametrize(
    "url,body",
    [
        (LOGIN_URL, "Incorrect response."),
        ("https://store.epicgames.com/en-US/free-games", REJECTION),
    ],
)
def test_rejection_detection_requires_the_full_login_page_message(login_state, url, body):
    auth, _page, _clock = login_state(url=url, body=body)

    asyncio.run(auth._raise_visible_login_rejection())


def test_captcha_success_alone_does_not_authenticate(auth_module, login_state, monkeypatch):
    auth, _page, clock = login_state()
    solver = AsyncMock()
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solver)

    with pytest.raises(auth_module.PlaywrightTimeoutError):
        asyncio.run(
            auth._await_login_outcome(
                LOGIN_URL, object(), timeout_seconds=4, challenge_succeeded=True
            )
        )

    solver.assert_not_awaited()
    assert clock.now == 104.0


def test_mfa_submission_can_extend_short_wait_for_real_login_signal(
    auth_module, login_state, monkeypatch
):
    auth, _page, clock = login_state(url=f"{LOGIN_URL}/mfa")
    solver = AsyncMock()
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solver)
    clock.on_wait = lambda: (
        auth._is_login_success_signal.put_nowait({"accountId": "offline-account"})
        if clock.now >= 105.0
        else None
    )

    asyncio.run(auth._await_login_outcome(LOGIN_URL, object(), timeout_seconds=1))

    auth_module.submit_totp_challenge.assert_awaited_once()
    solver.assert_not_awaited()
    assert clock.now == 105.0


def test_repeated_mfa_challenges_cannot_extend_the_hard_deadline(
    auth_module, login_state, monkeypatch
):
    auth, _page, clock = login_state(url=f"{LOGIN_URL}/mfa", frames=[_active_frame()])
    solver = AsyncMock(return_value=ChallengeSignal.SUCCESS)
    monkeypatch.setattr(auth_module, "wait_for_challenge_signal", solver)

    with pytest.raises(auth_module.PlaywrightTimeoutError):
        asyncio.run(auth._await_login_outcome(LOGIN_URL, object(), timeout_seconds=4))

    assert solver.await_count > 1
    assert clock.now <= 280.5


def _response(url, *, payload=None, status=200, method="POST", non_json=False):
    return SimpleNamespace(
        url=url,
        status=status,
        request=SimpleNamespace(method=method),
        json=AsyncMock(
            side_effect=ValueError("private-body") if non_json else None, return_value=payload
        ),
        text=AsyncMock(side_effect=AssertionError("Raw response body must not be read")),
    )


@pytest.mark.parametrize("non_json", [False, True])
def test_login_api_diagnostics_never_include_query_account_token_or_body(
    auth_module, login_state, non_json
):
    auth, _page, _clock = login_state()
    response = _response(
        "https://www.epicgames.com/id/api/login?token=private-token&account=private-account",
        status=503 if non_json else 400,
        payload={
            "errorCode": "errors.com.epicgames.accountportal.captcha_invalid",
            "accountId": "private-account",
            "message": "private-body",
            "token": "private-token",
        },
        non_json=non_json,
    )

    asyncio.run(auth._on_response_anything(response))

    diagnostic_text = "\n".join(auth_module.test_messages)
    assert "login" in diagnostic_text
    assert str(response.status) in diagnostic_text
    assert "private-" not in diagnostic_text
    assert "?" not in diagnostic_text
    if not non_json:
        assert "errors.com.epicgames.accountportal.captcha_invalid" in diagnostic_text
    response.text.assert_not_awaited()
    assert auth._is_login_success_signal.empty()


@pytest.mark.parametrize(
    "unsafe_error_code",
    [
        "errors.com.epicgames.invalid?token=private-token",
        "errors.com.epicgames.invalid\nprivate-account",
        {"token": "private-token"},
        "private-body",
    ],
)
def test_untrusted_error_code_text_is_not_copied_to_diagnostics(
    auth_module, login_state, unsafe_error_code
):
    auth, _page, _clock = login_state()
    response = _response(
        "https://www.epicgames.com/id/api/login",
        status=400,
        payload={"errorCode": unsafe_error_code},
    )

    asyncio.run(auth._on_response_anything(response))

    diagnostic_text = "\n".join(auth_module.test_messages)
    assert "unknown" in diagnostic_text
    assert "private-" not in diagnostic_text
    with pytest.raises(RuntimeError) as raised:
        asyncio.run(auth._await_login_outcome(LOGIN_URL, object()))
    assert "private-" not in str(raised.value)
    assert "unknown" in str(raised.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.epicgames.com.evil.invalid/id/api/analytics",
        "https://evil.invalid/id/api/login?next=https://www.epicgames.com",
        "https://www.epicgames.com@evil.invalid/id/api/analytics",
        "http://www.epicgames.com/id/api/analytics",
        "https://www.epicgames.com/unrelated?next=/id/api/analytics",
    ],
)
def test_untrusted_hosts_and_query_matches_do_not_emit_auth_signals_or_diagnostics(
    auth_module, login_state, url
):
    auth, _page, _clock = login_state()
    response = _response(
        url, payload={"accountId": "private-account", "errorCode": "private-error"}
    )

    asyncio.run(auth._on_response_anything(response))

    assert auth._is_login_success_signal.empty()
    assert auth._login_error_signal.empty()
    assert auth_module.test_messages == []
    response.json.assert_not_awaited()


@pytest.mark.parametrize(
    "endpoint,payload,expected_success",
    [
        ("login", {"success": True}, False),
        ("analytics", {"success": True}, False),
        ("analytics", {"accountId": "offline-account"}, True),
    ],
)
def test_only_existing_authenticated_api_signal_counts_as_login_success(
    login_state, endpoint, payload, expected_success
):
    auth, _page, _clock = login_state()

    asyncio.run(
        auth._on_response_anything(
            _response(f"https://www.epicgames.com/id/api/{endpoint}", payload=payload)
        )
    )

    assert (not auth._is_login_success_signal.empty()) is expected_success
