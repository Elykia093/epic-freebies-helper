# -*- coding: utf-8 -*-
import asyncio
from contextlib import asynccontextmanager, suppress

import pytest

from extensions.llm_errors import (
    LLMConfigurationError,
    LLMRequestAbort,
    llm_request_boundary,
)


def test_abort_bypasses_exception_recovery_and_converts_at_boundary():
    recovered = []

    async def challenge():
        with suppress(Exception):
            try:
                raise LLMRequestAbort("LLM request rejected: HTTP 401")
            except Exception:
                recovered.append(True)
        recovered.append("continued")

    async def run():
        async with llm_request_boundary():
            await asyncio.wait_for(challenge(), timeout=1)

    with pytest.raises(LLMConfigurationError, match="HTTP 401") as caught:
        asyncio.run(run())

    assert recovered == []
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_browser_cleanup_finishes_before_abort_is_converted():
    events = []

    @asynccontextmanager
    async def browser_context():
        events.append("opened")
        try:
            yield
        finally:
            await asyncio.sleep(0)
            events.append("closed")

    async def run():
        try:
            async with llm_request_boundary(), browser_context():
                raise LLMRequestAbort("LLM request rejected: HTTP 400")
        except LLMConfigurationError:
            events.append("converted")
            raise

    with pytest.raises(LLMConfigurationError, match="HTTP 400"):
        asyncio.run(run())

    assert events == ["opened", "closed", "converted"]


@pytest.mark.parametrize("error_type", [TimeoutError, RuntimeError])
def test_boundary_preserves_ordinary_errors(error_type):
    error = error_type("retryable failure")

    async def run():
        async with llm_request_boundary():
            raise error

    with pytest.raises(error_type) as caught:
        asyncio.run(run())

    assert caught.value is error


def test_boundary_preserves_task_cancellation_and_cleanup():
    events = []

    @asynccontextmanager
    async def browser_context():
        try:
            yield
        finally:
            events.append("closed")

    async def run():
        entered = asyncio.Event()

        async def worker():
            async with llm_request_boundary(), browser_context():
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    asyncio.run(run())
    assert events == ["closed"]
