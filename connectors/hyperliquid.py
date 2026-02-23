"""
connectors/hyperliquid.py
=========================
Hyperliquid connector.
Uses the single public info endpoint: https://api.hyperliquid.xyz/info

Funding interval: 1 hour (3600 seconds) for all markets.

Symbol canonicalisation: BTC → BTC (already canonical, no stripping needed).

Key endpoints (all POST to /info):
  {"type": "metaAndAssetCtxs"}          → all meta + live market data in one shot
  {"type": "l2Book", "coin": "BTC"}     → orderbook  (levels[0]=bids, levels[1]=asks)
  {"type": "fundingHistory", "coin": "BTC", "startTime": <ms>} → historical funding
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

from core.models import MarketSnapshot, OrderbookLevel, Venue

log = logging.getLogger(__name__)

BASE_URL   = "https://api.hyperliquid.xyz/info"
VENUE      = Venue.HYPERLIQUID
INTERVAL_S = 3600   # 1-hour funding



def canonicalize(raw: str) -> str:
    """BTC → BTC (Hyperliquid symbols are already canonical)."""
    return raw.upper()



async def fetch_meta_and_asset_ctxs(
    client: httpx.AsyncClient,
) -> Tuple[List[Dict], List[Dict]]:
    """
    POST {"type": "metaAndAssetCtxs"}
    Returns (universe, assetCtxs) where universe[i] <-> assetCtxs[i].

    universe[i] fields: name, szDecimals, maxLeverage, onlyIsolated
    assetCtxs[i] fields: funding, openInterest, prevDayPx, dayNtlVlm,
                         premium, oraclePx, markPx, midPx, impactPxs, ...
    """
    resp = await client.post(
        BASE_URL,
        json={"type": "metaAndAssetCtxs"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    # data is [universe_list, asset_ctx_list]
    universe   = data[0]["universe"]
    asset_ctxs = data[1]
    return universe, asset_ctxs


async def fetch_orderbook(
    client: httpx.AsyncClient,
    coin: str,
    depth: int = 20,
) -> Tuple[List[OrderbookLevel], List[OrderbookLevel]]:
    """
    POST {"type": "l2Book", "coin": "BTC", "nSigFigs": 5}
    Response: {"coin": "BTC", "levels": [[bid_levels], [ask_levels]], "time": <ms>}
    Each level: {"px": "...", "sz": "...", "n": <count>}

    Returns (bids_desc, asks_asc).
    """
    resp = await client.post(
        BASE_URL,
        json={"type": "l2Book", "coin": coin, "nSigFigs": 5},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    raw_bids = data["levels"][0]  # bids: best bid first (descending)
    raw_asks = data["levels"][1]  # asks: best ask first (ascending)

    # Limit to requested depth
    raw_bids = raw_bids[:depth]
    raw_asks = raw_asks[:depth]

    bids = [OrderbookLevel(price=float(b["px"]), size=float(b["sz"])) for b in raw_bids]
    asks = [OrderbookLevel(price=float(a["px"]), size=float(a["sz"])) for a in raw_asks]

    bids.sort(key=lambda x: x.price, reverse=True)  # best bid first
    asks.sort(key=lambda x: x.price)                 # best ask first

    return bids, asks


async def fetch_funding_history(
    client: httpx.AsyncClient,
    coin: str,
    hours_back: int = 168,   # 7 days default
) -> List[Dict]:
    """
    POST {"type": "fundingHistory", "coin": "BTC", "startTime": <ms>}
    Returns list of {coin, fundingRate, premium, time}.
    Most recent last.
    """
    start_ms = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp() * 1000)
    resp = await client.post(
        BASE_URL,
        json={"type": "fundingHistory", "coin": coin, "startTime": start_ms},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()



async def fetch_market_snapshot(
    client: httpx.AsyncClient,
    coin: str,
    asset_ctx: Dict,
    depth: int = 20,
) -> MarketSnapshot:
    """
    Fetch orderbook and combine with pre-fetched asset_ctx to build MarketSnapshot.
    asset_ctx comes from metaAndAssetCtxs (avoids per-market API call for price data).

    asset_ctx fields used:
      funding   - current hourly funding rate (string)
      oraclePx  - index/oracle price
      markPx    - mark price
      openInterest - OI in coin units (multiply by markPx for USD notional)
    """
    bids, asks = await fetch_orderbook(client, coin, depth)

    funding_rate = float(asset_ctx.get("funding", 0))
    oracle_px    = float(asset_ctx.get("oraclePx", 0))
    mark_px_str  = asset_ctx.get("markPx")
    mark_price   = float(mark_px_str) if mark_px_str else oracle_px

    oi_coin = float(asset_ctx.get("openInterest", 0))
    oi_usd  = oi_coin * mark_price

    # Hyperliquid doesn't expose nextFundingTime directly;
    # funding settles at the top of each hour.
    now = datetime.utcnow()
    next_funding = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    return MarketSnapshot(
        venue=VENUE,
        symbol=canonicalize(coin),
        market=coin,
        ts=now,
        mark_price=mark_price,
        index_price=oracle_px,
        funding_rate=funding_rate,
        funding_interval_seconds=INTERVAL_S,
        open_interest_usd=oi_usd,
        orderbook_bids=bids,
        orderbook_asks=asks,
        next_funding_time=next_funding,
        orderbook_stale=False,
    )



async def fetch_all_snapshots(
    client: httpx.AsyncClient,
    markets: List[str],
    depth: int = 20,
) -> List[MarketSnapshot]:
    """
    Fetch snapshots for all configured markets.
    Strategy:
      1. Single metaAndAssetCtxs call → all price/funding data at once (efficient)
      2. Parallel orderbook fetches for each market
    Failed markets are logged and skipped.
    """
    # Step 1: one call gets all price/funding data
    try:
        universe, asset_ctxs = await fetch_meta_and_asset_ctxs(client)
    except Exception as exc:
        log.error("Hyperliquid metaAndAssetCtxs failed: %s", exc)
        return []

    # Build coin → asset_ctx lookup
    ctx_by_coin: Dict[str, Dict] = {}
    for meta, ctx in zip(universe, asset_ctxs):
        ctx_by_coin[meta["name"]] = ctx

    # Step 2: parallel orderbook fetch for each requested market
    async def _fetch_one(coin: str) -> Optional[MarketSnapshot]:
        ctx = ctx_by_coin.get(coin)
        if ctx is None:
            log.warning("Hyperliquid: %s not found in metaAndAssetCtxs", coin)
            return None
        try:
            return await fetch_market_snapshot(client, coin, ctx, depth)
        except Exception as exc:
            log.warning("Hyperliquid fetch failed for %s: %s", coin, exc)
            return None

    tasks = [_fetch_one(coin) for coin in markets]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]



async def fetch_orderbook_only(
    client: httpx.AsyncClient,
    coin: str,
    depth: int = 20,
) -> Tuple[List[OrderbookLevel], List[OrderbookLevel]]:
    """Lightweight orderbook-only fetch for high-frequency polling."""
    return await fetch_orderbook(client, coin, depth)



async def fetch_active_markets(client: httpx.AsyncClient) -> List[str]:
    """
    Returns list of all active perpetual coin names on Hyperliquid.
    Uses metaAndAssetCtxs; filters out any with zero mark price (inactive).
    """
    universe, asset_ctxs = await fetch_meta_and_asset_ctxs(client)
    active = []
    for meta, ctx in zip(universe, asset_ctxs):
        try:
            mark = float(ctx.get("markPx") or 0)
            if mark > 0:
                active.append(meta["name"])
        except (ValueError, TypeError):
            pass
    return active
