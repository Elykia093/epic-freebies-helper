# -*- coding: utf-8 -*-
"""Stop unrecoverable LLM requests without triggering captcha recovery loops."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class LLMRequestAbort(BaseException):
    """Internal control signal that bypasses third-party ``except Exception`` retries.

    Keep this separate from cancellation and convert it at the outer request boundary
    so callers and notification handlers receive a normal application exception.
    """

    def __init__(self, message: str, *, is_configuration_error: bool = False):
        super().__init__(message)
        self.is_configuration_error = is_configuration_error


class LLMConfigurationError(RuntimeError):
    """An LLM request cannot succeed without changing its configuration."""


class LLMResponseError(ValueError):
    """A response was received, but cannot safely be used as a challenge answer."""


@asynccontextmanager
async def llm_request_boundary() -> AsyncIterator[None]:
    """Convert abort signals after inner browser contexts have finished cleanup."""
    try:
        yield
    except LLMRequestAbort as err:
        error_type = LLMConfigurationError if err.is_configuration_error else RuntimeError
        raise error_type(str(err)) from None
