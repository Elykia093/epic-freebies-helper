# -*- coding: utf-8 -*-
from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from models import PromotionGame
from services.epic_games_service import EpicAgent, get_promotions


class CollectionSummary(BaseModel):
    all_promotions: list[PromotionGame] = Field(default_factory=list)
    newly_claimed_promotions: list[PromotionGame] = Field(default_factory=list)
    previously_claimed_promotions: list[PromotionGame] = Field(default_factory=list)
    failed_promotions: list[PromotionGame] = Field(default_factory=list)
    error_message: str = ""


class EpicCollectionSummaryError(RuntimeError):
    def __init__(self, message: str, summary: CollectionSummary):
        super().__init__(message)
        self.summary = summary


def _promotion_key(promotion: PromotionGame) -> str:
    return promotion.namespace or promotion.id or promotion.url


def _unique_promotions(promotions: list[PromotionGame]) -> list[PromotionGame]:
    result: list[PromotionGame] = []
    keys: set[str] = set()
    for promotion in promotions:
        key = _promotion_key(promotion)
        if key in keys:
            continue
        result.append(promotion)
        keys.add(key)
    return result


def _promotions_in_namespaces(
    promotions: list[PromotionGame], namespaces: set[str]
) -> list[PromotionGame]:
    return _unique_promotions(
        [promotion for promotion in promotions if promotion.namespace in namespaces]
    )


async def _refresh_order_snapshot(agent: EpicAgent) -> set[str]:
    agent._orders = []
    agent._namespaces = []
    await agent._sync_order_history()
    await agent._check_orders()
    return set(agent._namespaces)


async def collect_epic_games_with_summary(agent: EpicAgent) -> CollectionSummary:
    all_promotions = get_promotions()
    before_namespaces = await _refresh_order_snapshot(agent)
    previously_claimed = _promotions_in_namespaces(all_promotions, before_namespaces)
    pending_promotions = _unique_promotions(
        [promotion for promotion in all_promotions if promotion.namespace not in before_namespaces]
    )

    try:
        await agent.collect_epic_games()
    except Exception as err:
        summary = CollectionSummary(
            all_promotions=all_promotions,
            previously_claimed_promotions=previously_claimed,
            failed_promotions=pending_promotions,
            error_message=str(err),
        )
        raise EpicCollectionSummaryError(str(err), summary) from err

    try:
        after_namespaces = await _refresh_order_snapshot(agent)
    except Exception as err:
        logger.warning(
            "Failed to refresh Epic order history after collection | error_type={}",
            type(err).__name__,
        )
        message = f"Failed to refresh Epic order history after collection: {type(err).__name__}"
        summary = CollectionSummary(
            all_promotions=all_promotions,
            previously_claimed_promotions=previously_claimed,
            failed_promotions=pending_promotions,
            error_message=message,
        )
        raise EpicCollectionSummaryError(message, summary) from err

    newly_claimed = _promotions_in_namespaces(all_promotions, after_namespaces - before_namespaces)
    failed_promotions = _unique_promotions(
        [
            promotion
            for promotion in pending_promotions
            if promotion.namespace and promotion.namespace not in after_namespaces
        ]
    )

    return CollectionSummary(
        all_promotions=all_promotions,
        newly_claimed_promotions=newly_claimed,
        previously_claimed_promotions=previously_claimed,
        failed_promotions=failed_promotions,
    )
