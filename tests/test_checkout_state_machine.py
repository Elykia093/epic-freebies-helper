import asyncio
from types import SimpleNamespace

import pytest

import services.epic_games_service as epic_games_service
from services.epic_games_service import EpicGames


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    async def wait(self, timeout_ms):
        self.value += timeout_ms / 1000


class MissingLocator:
    @property
    def first(self):
        return self

    async def is_visible(self, timeout):
        return False


class SlowBodyLocator:
    def __init__(self, clock, scans):
        self.clock = clock
        self.scans = scans

    async def inner_text(self, timeout):
        self.scans.append(timeout)
        self.clock.value += timeout / 1000
        return "CHECKOUT ADD TO LIBRARY"


class SlowCheckoutContainer:
    url = "https://store.epicgames.com/purchase#/free-checkout"

    def __init__(self, clock, scans):
        self.clock = clock
        self.scans = scans

    def locator(self, selector, **kwargs):
        if selector == "body":
            return SlowBodyLocator(self.clock, self.scans)
        return MissingLocator()


class FakePage:
    def __init__(self, clock=None):
        self.clock = clock

    async def wait_for_timeout(self, timeout_ms):
        if self.clock is not None:
            await self.clock.wait(timeout_ms)


def test_active_purchase_container_uses_one_total_timeout(monkeypatch):
    clock = FakeClock()
    scans = []
    page = FakePage(clock)
    containers = [SlowCheckoutContainer(clock, scans) for _ in range(3)]

    async def ordered_containers(_page):
        return containers

    monkeypatch.setattr(epic_games_service.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(EpicGames, "_ordered_checkout_containers", staticmethod(ordered_containers))

    with pytest.raises(AssertionError, match="checkout submit button"):
        asyncio.run(
            EpicGames._active_purchase_container(
                page, place_order_timeout=500, confirm_timeout=500, log_missing=False
            )
        )

    assert scans == [500]
    assert clock.value == pytest.approx(0.5)


def test_observe_checkout_outcome_returns_pending_without_container(monkeypatch):
    clock = FakeClock()
    page = FakePage(clock)
    game = EpicGames(page)
    scans = 0

    async def no_device_modal(*args, **kwargs):
        return False

    async def not_visible(*args, **kwargs):
        return False

    async def missing_container(*args, **kwargs):
        nonlocal scans
        scans += 1
        clock.value += 2
        raise AssertionError("missing")

    monkeypatch.setattr(epic_games_service.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(game, "_handle_device_not_supported_modal", no_device_modal)
    monkeypatch.setattr(game, "_is_checkout_security_check_visible", not_visible)
    monkeypatch.setattr(game, "_is_claimed_state", not_visible)
    monkeypatch.setattr(game, "_active_purchase_container", missing_container)

    outcome = asyncio.run(
        game._observe_checkout_outcome(page, "https://example.test/game", timeout_ms=1000)
    )

    assert outcome == "pending"
    assert scans == 1


def test_security_clearance_requires_recoverable_checkout_state(monkeypatch):
    page = FakePage()
    game = EpicGames(page)
    visibility = iter([True, False])

    async def security_visibility(*args, **kwargs):
        return next(visibility)

    async def not_claimed(*args, **kwargs):
        return False

    async def pending_outcome(*args, **kwargs):
        return "pending"

    monkeypatch.setattr(game, "_is_checkout_security_check_visible", security_visibility)
    monkeypatch.setattr(game, "_is_claimed_state", not_claimed)
    monkeypatch.setattr(game, "_observe_checkout_outcome", pending_outcome)

    recovered = asyncio.run(
        game._resolve_checkout_security_check(
            page, object(), "https://example.test/game", max_wait_ms=1000
        )
    )

    assert recovered is False


def test_security_recovery_does_not_consume_submission_attempts(monkeypatch):
    page = FakePage()
    game = EpicGames(page)
    button = SimpleNamespace(text_content=lambda: None)
    purchase_payload = (object(), button)
    states = iter(
        [
            ("checkout", purchase_payload),
            ("security", None),
            ("checkout", purchase_payload),
            ("security", None),
            ("checkout", purchase_payload),
            ("security", None),
            ("checkout", purchase_payload),
        ]
    )
    submissions = 0
    security_resolutions = 0

    async def button_text_content():
        return "Add to library"

    button.text_content = button_text_content

    async def next_state(*args, **kwargs):
        return next(states)

    async def submit(*args, **kwargs):
        nonlocal submissions
        submissions += 1
        return True

    async def security_not_visible(*args, **kwargs):
        return False

    async def resolve_security(*args, **kwargs):
        nonlocal security_resolutions
        security_resolutions += 1
        return True

    async def probe(*args, **kwargs):
        return False

    async def observe(*args, **kwargs):
        return "claimed" if submissions == 4 else "security"

    async def unexpected_finalize(*args, **kwargs):
        pytest.fail("successful fourth submission must not enter final reconciliation")

    monkeypatch.setattr(epic_games_service, "AgentV", lambda **kwargs: object())
    monkeypatch.setattr(game, "_wait_for_purchase_state", next_state)
    monkeypatch.setattr(game, "_submit_place_order", submit)
    monkeypatch.setattr(game, "_is_checkout_security_check_visible", security_not_visible)
    monkeypatch.setattr(game, "_resolve_checkout_security_check", resolve_security)
    monkeypatch.setattr(game, "_probe_checkout_challenge", probe)
    monkeypatch.setattr(game, "_observe_checkout_outcome", observe)
    monkeypatch.setattr(game, "_finalize_unconfirmed_checkout", unexpected_finalize)

    claimed = asyncio.run(
        game._handle_instant_checkout(
            page,
            SimpleNamespace(url="https://example.test/game"),
            allow_finalize=False,
            timeout_ms=60000,
        )
    )

    assert claimed is True
    assert submissions == 4
    assert security_resolutions == 3


def test_initial_security_state_refreshes_checkout_before_submission(monkeypatch):
    page = FakePage()
    game = EpicGames(page)

    async def button_text_content():
        return "Add to library"

    button = SimpleNamespace(text_content=button_text_content)
    states = iter([("security", None), ("checkout", (object(), button))])
    submissions = 0

    async def next_state(*args, **kwargs):
        return next(states)

    async def resolve_security(*args, **kwargs):
        return True

    async def submit(*args, **kwargs):
        nonlocal submissions
        submissions += 1
        return True

    async def security_not_visible(*args, **kwargs):
        return False

    async def probe(*args, **kwargs):
        return False

    async def claimed_outcome(*args, **kwargs):
        return "claimed"

    monkeypatch.setattr(epic_games_service, "AgentV", lambda **kwargs: object())
    monkeypatch.setattr(game, "_wait_for_purchase_state", next_state)
    monkeypatch.setattr(game, "_resolve_checkout_security_check", resolve_security)
    monkeypatch.setattr(game, "_submit_place_order", submit)
    monkeypatch.setattr(game, "_is_checkout_security_check_visible", security_not_visible)
    monkeypatch.setattr(game, "_probe_checkout_challenge", probe)
    monkeypatch.setattr(game, "_observe_checkout_outcome", claimed_outcome)

    claimed = asyncio.run(
        game._handle_instant_checkout(
            page,
            SimpleNamespace(url="https://example.test/game"),
            allow_finalize=False,
            timeout_ms=60000,
        )
    )

    assert claimed is True
    assert submissions == 1
