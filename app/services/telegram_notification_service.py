# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from loguru import logger

from models import PromotionGame
from services.epic_collection_summary_service import CollectionSummary
from services.epic_games_service import get_promotions

RUN_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _github_run_url() -> str:
    repository = _env("GITHUB_REPOSITORY")
    run_id = _env("GITHUB_RUN_ID")
    if not repository or not run_id:
        return ""
    return f"https://github.com/{repository}/actions/runs/{run_id}"


def _format_error(error: Exception | str | None) -> str:
    if error is None:
        return "未知错误"

    message = str(error).strip() or type(error).__name__
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    for line in lines:
        lowered = line.lower()
        if lowered.startswith(("traceback", "file \"")):
            continue
        message = line
        break

    prefix = type(error).__name__ if isinstance(error, Exception) else ""
    if prefix and prefix not in message:
        message = f"{prefix}: {message}"

    if len(message) > 360:
        return message[:350] + "...(已截断)"
    return message


def _format_game_title(game: PromotionGame) -> str:
    return game.title or game.url or "Unknown"


def _format_games(games: list[PromotionGame]) -> str:
    if not games:
        return "无"

    lines = []
    for game in games:
        title = _format_game_title(game)
        url = game.url or ""
        lines.append(f"- {title}" + (f"\n  {url}" if url else ""))
    return "\n".join(lines)


def _runtime_lines() -> list[str]:
    run_time = datetime.now(RUN_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"运行时间：{run_time} Asia/Shanghai",
    ]

    if workflow := _env("GITHUB_WORKFLOW"):
        lines.append(f"工作流：{workflow}")

    if event_name := _env("GITHUB_EVENT_NAME"):
        lines.append(f"触发方式：{event_name}")

    if ref_name := _env("GITHUB_REF_NAME"):
        lines.append(f"分支：{ref_name}")

    if sha := _env("GITHUB_SHA"):
        lines.append(f"提交：{sha[:7]}")

    run_number = _env("GITHUB_RUN_NUMBER")
    run_attempt = _env("GITHUB_RUN_ATTEMPT")
    if run_number:
        suffix = f" / attempt {run_attempt}" if run_attempt else ""
        lines.append(f"运行编号：#{run_number}{suffix}")

    if run_url := _github_run_url():
        lines.extend(["运行链接：", run_url])

    return lines


def build_telegram_run_message(success: bool, error: Exception | str | None = None) -> str:
    summary = CollectionSummary(error_message=_format_error(error) if not success else "")
    return build_telegram_summary_message(summary)


def build_telegram_summary_message(summary: CollectionSummary) -> str:
    success = not summary.error_message
    sections = [
        "Epic 周免领取结果",
        "",
        f"运行状态：{'成功' if success else '失败'}",
        *_runtime_lines(),
        "",
        "本周游戏：",
        _format_games(summary.all_promotions),
        "",
        "本次新领取：",
        _format_games(summary.newly_claimed_promotions),
        "",
        "之前已领取：",
        _format_games(summary.previously_claimed_promotions),
    ]

    if summary.failed_promotions:
        sections.extend(["", "未确认成功：", _format_games(summary.failed_promotions)])

    if summary.error_message:
        sections.extend(["", "失败原因：", _format_error(summary.error_message)])

    message = "\n".join(sections)
    if len(message) > 3900:
        return message[:3890] + "\n...(内容过长已截断)"
    return message


def _safe_current_promotions() -> list[PromotionGame]:
    try:
        return get_promotions()
    except Exception as err:
        logger.warning(
            "Failed to load current Epic promotions for failure notification | error_type={}",
            type(err).__name__,
        )
        return []


def failure_summary_from_exception(err: Exception) -> CollectionSummary:
    summary = getattr(err, "summary", None)
    if not isinstance(summary, CollectionSummary):
        promotions = _safe_current_promotions()
        summary = CollectionSummary(
            all_promotions=promotions,
            failed_promotions=promotions,
        )

    if not summary.error_message:
        summary.error_message = _format_error(err)
    return summary


async def send_collection_summary_to_telegram(summary: CollectionSummary) -> None:
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.debug("Telegram notification is not configured; skipping delivery")
        return

    payload: dict[str, object] = {
        "chat_id": chat_id,
        "text": build_telegram_summary_message(summary),
        "disable_web_page_preview": True,
    }

    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as err:
        logger.warning("Telegram notification failed | status={}", err.response.status_code)
        return
    except httpx.HTTPError as err:
        logger.warning("Telegram notification failed | error_type={}", type(err).__name__)
        return

    logger.success("Telegram claim summary sent")
