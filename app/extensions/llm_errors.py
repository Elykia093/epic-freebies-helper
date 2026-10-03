# -*- coding: utf-8 -*-
"""Stop unrecoverable LLM requests without triggering captcha recovery loops."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class LLMRequestAbort(BaseException):
    """Internal control signal that bypasses third-party ``except Exception`` retries.

    Keep this separate from cancellation and convert it at the outer request boundary
    so callers and notification handlers receive a normal application exception.
    """


class LLMConfigurationError(RuntimeError):
    """An LLM request cannot succeed without changing its configuration."""


@asynccontextmanager
async def llm_request_boundary() -> AsyncIterator[None]:
    """Convert abort signals after inner browser contexts have finished cleanup."""
    try:
        yield
    except LLMRequestAbort as err:
        raise LLMConfigurationError(str(err)) from None
