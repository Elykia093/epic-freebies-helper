"""Offline regressions for recovery after Epic returns to the email form."""

from __future__ import annotations

import ast
import asyncio
import re
from contextlib import suppress
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urljoin, urlsplit

import pytest
from playwright._impl._errors import Error as PlaywrightError
from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError


CAPTCHA_INVALID = "errors.com.epicgames.accountportal.captcha_invalid"


class ChallengeSignal(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@pytest.fixture
def recovery():
    """Run the real class without importing settings or reading account secrets."""
    source = Path(__file__).parents[1] / "app/services/epic_authorization_service.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    clock = [0.0]
    state = SimpleNamespace(
        stage="email",
        challenge=False,
        challenge_after_email=False,
        success_after_tick=False,
        failure_phase=None,
    )
    actions = []
    namespace = {
        "asyncio": asyncio,
        "re": re,
        "suppress": suppress,
        "parse_qs": parse_qs,
        "urljoin": urljoin,
        "urlsplit": urlsplit,
        "time": SimpleNamespace(monotonic=lambda: clock[0]),
        "logger": Mock(),
        "LOGIN_CAPTCHA_ATTEMPT_LIMIT": 3,
        "ChallengeSignal": ChallengeSignal,
        "PlaywrightError": PlaywrightError,
        "PlaywrightTimeoutError": PlaywrightTimeoutError,
        "settings": SimpleNamespace(
            EXECUTION_TIMEOUT=120, RESPONSE_TIMEOUT=30, EPIC_EMAIL="offline@example.invalid"
        ),
        "totp_login_enabled": lambda: True,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)

    class Locator:
        def __init__(self, selector):
            self.selector = selector

        async def is_visible(self, **_kwargs):
            return (
                self.selector
                in {
                    "email": {"#email", "#continue"},
                    "password": {"#password", "#sign-in"},
                    "mfa": set(),
                    "missing": set(),
                }[state.stage]
            )

        async def count(self):
            return int(await self.is_visible())

        async def wait_for(self, *, state="visible", timeout=1000):
            assert state == "visible"
            if not await self.is_visible():
                clock[0] += timeout / 1000
                raise PlaywrightTimeoutError("Offline control is unavailable")

        async def fill(self, value, **_kwargs):
            assert self.selector == "#email"
            assert value == "offline@example.invalid"
            actions.append("fill_email")

        async def click(self, **_kwargs):
            assert self.selector == "#continue"
            actions.append("continue")
            state.stage = "password"

    class Page:
        url = "https://www.epicgames.com/id/login"
        frames = []

        def locator(self, selector):
            return Locator(selector)

        async def reload(self, **_kwargs):
            assert self.url == "https://www.epicgames.com/id/login"
            actions.append("reload")
            if state.failure_phase == "reload":
                raise PlaywrightTimeoutError("Offline page refresh timed out")
            if state.failure_phase == "controls":
                state.stage = "missing"

        async def wait_for_timeout(self, milliseconds):
            clock[0] += milliseconds / 1000
            if state.success_after_tick:
                auth._is_login_success_signal.put_nowait({"authenticated": True})
                state.success_after_tick = False
            await asyncio.sleep(0)

    auth = namespace["EpicAuthorization"](Page())
    auth._email_step_completed = True
    auth._has_active_hcaptcha_challenge = AsyncMock(side_effect=lambda: state.challenge)

    async def submit_email(**_kwargs):
        if state.stage != "email":
            return False
        assert not state.challenge, "An active challenge must settle before form submission"
        actions.append("email")
        if state.challenge_after_email:
            state.challenge = True
        else:
            state.stage = "password"
        return True

    async def submit_password(**_kwargs):
        assert not state.challenge, "Password submission must wait for the email challenge"
        if state.stage != "password":
            return False
        actions.append("password")
        auth._is_login_success_signal.put_nowait({"authenticated": True})
        return True

    async def solve(_agent, **_kwargs):
        assert state.challenge
        actions.append("solve")
        clock[0] += 0.5
        state.challenge = False
        state.stage = "password"
        return ChallengeSignal.SUCCESS

    auth._resubmit_email_form = AsyncMock(side_effect=submit_email)
    auth._resubmit_password_form = AsyncMock(side_effect=submit_password)
    namespace["wait_for_challenge_signal"] = solve

    async def reject_email():
        response = SimpleNamespace(
            url="https://www.epicgames.com/id/api/email",
            request=SimpleNamespace(method="POST"),
            json=AsyncMock(return_value={"errorCode": CAPTCHA_INVALID}),
        )
        await auth._on_response_anything(response)

    async def wait():
        return await auth._await_login_outcome(auth.page.url, object(), timeout_seconds=25)

    return SimpleNamespace(
        auth=auth, state=state, actions=actions, clock=clock, reject_email=reject_email, wait=wait
    )


def test_email_rejection_recovers_current_form_despite_previous_password_step(recovery):
    async def run():
        await recovery.reject_email()
        await recovery.wait()

    asyncio.run(run())

    assert recovery.actions == ["email", "password"]
    assert recovery.auth._captcha_rejection_retries == 1
    assert not recovery.auth._email_recovery_pending
    assert recovery.auth._email_step_completed
    assert recovery.clock[0] < 25
    recovery.auth._resubmit_password_form.assert_awaited_once_with(timeout_ms=1000)


def test_email_challenge_is_solved_before_recovered_password_is_submitted(recovery):
    recovery.state.challenge = True

    async def run():
        await recovery.reject_email()
        await recovery.wait()

    asyncio.run(run())

    assert recovery.actions == ["solve", "password"]
    assert recovery.auth._captcha_solve_attempts == 1
    assert recovery.auth._captcha_rejection_retries == 1
    recovery.auth._resubmit_email_form.assert_not_awaited()
    recovery.auth._resubmit_password_form.assert_awaited_once_with(timeout_ms=1000)


def test_resubmitted_email_challenge_is_completed_before_password_submission(recovery):
    recovery.state.challenge_after_email = True

    async def run():
        await recovery.reject_email()
        await recovery.wait()

    asyncio.run(run())

    assert recovery.actions == ["email", "solve", "password"]
    assert recovery.auth._captcha_solve_attempts == 1
    assert recovery.auth._captcha_rejection_retries == 1
    assert not recovery.auth._email_recovery_pending


def test_email_rejection_obeys_shared_recovery_budget(recovery):
    recovery.auth._captcha_rejection_retries = 3

    async def run():
        await recovery.reject_email()
        await recovery.wait()

    with pytest.raises(RuntimeError, match="captcha_invalid"):
        asyncio.run(run())

    assert recovery.clock[0] == 0
    assert recovery.actions == []
    recovery.auth._resubmit_email_form.assert_not_awaited()
    recovery.auth._resubmit_password_form.assert_not_awaited()


def test_email_rejection_without_controls_or_challenge_fails_explicitly(recovery):
    recovery.state.stage = "missing"

    async def run():
        await recovery.reject_email()
        await recovery.wait()

    with pytest.raises(RuntimeError, match="captcha_invalid"):
        asyncio.run(run())

    assert recovery.clock[0] < 25
    assert recovery.actions == []


@pytest.mark.parametrize("progress", ["success", "mfa"])
def test_email_rejection_does_not_restart_an_advanced_login(recovery, progress):
    if progress == "success":
        recovery.auth._is_login_success_signal.put_nowait({"authenticated": True})
    else:
        recovery.state.stage = "mfa"
        recovery.auth.page.url = "https://www.epicgames.com/id/login/mfa"
        recovery.auth._totp_attempts = 1
        recovery.state.success_after_tick = True

    async def run():
        await recovery.reject_email()
        await recovery.wait()

    asyncio.run(run())

    assert recovery.actions == []
    assert recovery.auth._captcha_rejection_retries == 0
    recovery.auth._resubmit_email_form.assert_not_awaited()
    recovery.auth._resubmit_password_form.assert_not_awaited()


def test_real_email_resubmission_refreshes_the_same_page_before_continuing(recovery):
    result = asyncio.run(type(recovery.auth)._resubmit_email_form(recovery.auth))

    assert result is True
    assert recovery.state.stage == "password"
    assert recovery.actions == ["reload", "fill_email", "continue"]


@pytest.mark.parametrize("stage", ["password", "missing", "mfa"])
def test_real_email_resubmission_does_not_refresh_other_stages(recovery, stage):
    recovery.state.stage = stage

    result = asyncio.run(type(recovery.auth)._resubmit_email_form(recovery.auth))

    assert result is False
    assert recovery.state.stage == stage
    assert recovery.actions == []


@pytest.mark.parametrize("phase", ["reload", "controls"])
def test_real_email_resubmission_stops_after_refresh_or_control_timeout(recovery, phase):
    recovery.state.failure_phase = phase

    result = asyncio.run(type(recovery.auth)._resubmit_email_form(recovery.auth))

    assert result is False
    assert recovery.actions == ["reload"]
    recovery.auth._resubmit_password_form.assert_not_awaited()
