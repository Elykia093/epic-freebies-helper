# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
from contextlib import suppress

import pyotp
from loguru import logger
from playwright.async_api import Page, expect


def _totp_secret_value() -> str | None:
    secret = os.getenv("EPIC_TOTP_SECRET", "").replace(" ", "").strip()
    return secret or None


async def _current_totp_code(page: Page) -> str | None:
    secret = _totp_secret_value()
    if not secret:
        return None

    try:
        totp = pyotp.TOTP(secret)
        remaining = totp.interval - (time.time() % totp.interval)
        if remaining < 5:
            await page.wait_for_timeout(int((remaining + 1) * 1000))
        return totp.now()
    except Exception as err:
        logger.error(
            "Failed to generate Epic authenticator TOTP code; EPIC_TOTP_SECRET is likely "
            "not a valid base32 secret | error_type={}",
            type(err).__name__,
        )
        return None


async def _select_authenticator_mfa_method(page: Page) -> None:
    with suppress(Exception):
        clicked = await page.evaluate(
            """
            () => {
              const normalize = (value) =>
                (value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const isVisible = (element) => {
                const rect = element.getBoundingClientRect();
                const style = window.getComputedStyle(element);
                return rect.width > 0 && rect.height > 0 &&
                  style.visibility !== 'hidden' &&
                  style.display !== 'none' &&
                  style.opacity !== '0';
              };
              const preferred = ['authenticator', 'authentication app', 'verification app'];
              const candidates = Array.from(document.querySelectorAll('button,a,label'))
                .filter(isVisible)
                .filter((element) => {
                  const text = normalize(element.innerText || element.textContent);
                  return preferred.some((marker) => text.includes(marker));
                });
              const target = candidates[0];
              if (!target) {
                return false;
              }
              target.click();
              return true;
            }
            """
        )
        if clicked:
            await page.wait_for_timeout(1000)


async def submit_totp_challenge(page: Page) -> bool:
    code = await _current_totp_code(page)
    if not code:
        logger.error(
            "Epic account requires authenticator 2FA, but no valid TOTP code could be "
            "generated. Set EPIC_TOTP_SECRET to a valid base32 authenticator secret."
        )
        return False

    await _select_authenticator_mfa_method(page)

    selectors = (
        "input[autocomplete='one-time-code']",
        "input[name='code']",
        "input[id*='code']",
        "input[inputmode='numeric']",
        "input[type='tel']",
        "input[type='text']",
    )

    filled = False
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = await locator.count()
            if not count:
                continue
            if count >= 6:
                for index, digit in enumerate(code):
                    await locator.nth(index).fill(digit, timeout=1000)
            else:
                field = locator.first
                await expect(field).to_be_visible(timeout=2000)
                await field.fill(code, timeout=2000)
            filled = True
            break
        except Exception:
            continue

    if not filled:
        logger.error("Could not find Epic authenticator 2FA code input")
        return False

    clicked = False
    for selector in (
        "#continue",
        "#sign-in",
        "button[type='submit']",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'verify')]",
    ):
        with suppress(Exception):
            button = page.locator(selector).first
            if await button.is_visible(timeout=1000):
                await button.click(timeout=2000)
                clicked = True
                break

    if not clicked:
        with suppress(Exception):
            await page.keyboard.press("Enter")
            clicked = True

    if clicked:
        logger.success("Submitted Epic authenticator 2FA code")
        await page.wait_for_timeout(1500)
    return clicked
