import asyncio
from types import SimpleNamespace

import pytest
from camoufox.fingerprints import Screen

import services.browser_context as browser_context
from services.browser_context import _install_hsw_identity_route


def test_hsw_route_forces_identity_without_dropping_request_headers():
    class Request:
        @staticmethod
        async def all_headers():
            return {"accept-encoding": "gzip, deflate", "x-request-id": "kept"}

    class Route:
        request = Request()
        continued_headers = None

        async def continue_(self, *, headers):
            self.continued_headers = headers

    class Context:
        pattern = None
        handler = None

        async def route(self, pattern, handler):
            self.pattern = pattern
            self.handler = handler

    async def scenario():
        context = Context()
        route = Route()
        await _install_hsw_identity_route(context)
        await context.handler(route)
        return context, route

    context, route = asyncio.run(scenario())

    assert context.pattern == "**/hsw.js*"
    assert route.continued_headers == {"accept-encoding": "identity", "x-request-id": "kept"}


@pytest.mark.parametrize("headless", [True, False, "virtual"])
def test_camoufox_options_use_paired_library_screen_and_preserve_context_options(
    monkeypatch, tmp_path, headless
):
    profile = tmp_path / "profile"
    recording = tmp_path / "recording"
    monkeypatch.setattr(
        browser_context, "settings", SimpleNamespace(user_data_dir_for=lambda backend: profile)
    )
    monkeypatch.setattr(browser_context, "RECORD_DIR", recording)

    options = browser_context._camoufox_launch_options(headless, None)

    screen = options["screen"]
    assert isinstance(screen, Screen)
    conditions = screen.as_conditions()
    assert conditions["screen.width"](1920)
    assert not conditions["screen.width"](1921)
    assert conditions["screen.height"](1080)
    assert not conditions["screen.height"](1079)
    assert options["persistent_context"] is True
    assert options["user_data_dir"] == profile
    assert options["record_video_dir"] == recording
    assert options["record_video_size"] == {"width": 1920, "height": 1080}
    assert options["headless"] == headless
    assert options["humanize"] == 0.2
    assert options["firefox_user_prefs"]["network.proxy.type"] == 0
    assert "proxy" not in options
    assert "geoip" not in options


def test_camoufox_options_keep_explicit_proxy_and_geoip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        browser_context,
        "settings",
        SimpleNamespace(user_data_dir_for=lambda backend: tmp_path / "profile"),
    )
    proxy = {"server": "http://127.0.0.1:8080"}

    options = browser_context._camoufox_launch_options(True, proxy)

    assert options["proxy"] is proxy
    assert options["geoip"] is True
    assert "network.proxy.type" not in options["firefox_user_prefs"]
