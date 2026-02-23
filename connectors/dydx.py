"""
connectors/dydx.py
==================
dYdX v4 connector.
Uses the public indexer API — no auth required.

Base URL: https://indexer.dydx.trade/v4

Funding interval: 1 hour (3600 seconds) for all markets.

Symbol canonicalisation: BTC-USD → BTC (split on '-', take first part).

Known data gap:
  dYdX v4 does not expose a separate markPrice. We use oraclePrice for both
  mark and index, so basis is always 0 for dYdX markets. This is documented
  in the UI and stored as-is (not imputed).

Funding note:
  nextFundingRate is the predicted payment for the NEXT settlement,
  expressed as a per-1-hour rate. We label this "predicted" in the UI.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

from core.models import MarketSnapshot, OrderbookLevel, Venue

log = logging.getLogger(__name__)

BASE_URL   = "https://indexer.dydx.trade/v4"
VENUE      = Venue.DYDX
INTERVAL_S = 3600   # 1-hour funding



def canonicalize(raw: str) -> str:
    """BTC-USD → BTC"""
    return raw.split("-")[0].upper()


def to_market(symbol: str) -> str:
    """BTC → BTC-USD (canonical → dYdX market ID)"""
    return f"{symbol.upper()}-USD"



async def fetch_perpetual_markets(
    client: httpx.AsyncClient,
) -> Dict[str, Dict]:
    """
    GET /perpetualMarkets
    Returns dict of {market_id: market_data} for all active perpetual markets.

    Key fields per market:
      status, oraclePrice, nextFundingRate, openInterest,
      baseOpenInterest, priceChange24H, volume24H, trades24H
    """
    resp = await client.get(
        f"{BASE_URL}/perpetualMarkets",
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("markets", {})


async def fetch_orderbook(
    client: httpx.AsyncClient,
    market: str,
    depth: int = 20,
) -> Tuple[List[OrderbookLevel], List[OrderbookLevel]]:
    """
    GET /orderbooks/perpetualMarket/{market}?limit=20
    Response: {"bids": [{"price": "...", "size": "..."}], "asks": [...]}

    Returns (bids_desc, asks_asc).
    """
    resp = await client.get(
        f"{BASE_URL}/orderbooks/perpetualMarket/{market}",
        params={"limit": depth},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    bids = [
        OrderbookLevel(price=float(b["price"]), size=float(b["size"]))
        for b in data.get("bids", [])
    ]
    asks = [
        OrderbookLevel(price=float(a["price"]), size=float(a["size"]))
        for a in data.get("asks", [])
    ]

    bids.sort(key=lambda x: x.price, reverse=True)  # best bid first
    asks.sort(key=lambda x: x.price)                 # best ask first

    return bids, asks


async def fetch_historical_funding(
    client: httpx.AsyncClient,
    market: str,
    limit: int = 100,
) -> List[Dict]:
    """
    GET /historicalFunding/{market}?limit=100
    Returns list of {ticker, rate, price, effectiveAt, effectiveAtHeight}.
    Most recent first.
    """
    resp = await client.get(
        f"{BASE_URL}/historicalFunding/{market}",
        params={"limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("historicalFunding", [])



async def fetch_market_snapshot(
    client: httpx.AsyncClient,
    market: str,
    market_data: Dict,
    depth: int = 20,
) -> MarketSnapshot:
    """
    Fetch orderbook and combine with pre-fetched market_data to build MarketSnapshot.
    market_data comes from /perpetualMarkets.

    Note: dYdX uses oraclePrice for both mark and index (basis = 0).
    nextFundingRate is the predicted per-hour rate for the next settlement.

    OI: baseOpenInterest (in coin units) × oraclePrice = USD notional.
    """
    bids, asks = await fetch_orderbook(client, market, depth)

    oracle_price   = float(market_data.get("oraclePrice", 0))
    funding_rate   = float(market_data.get("nextFundingRate", 0))

    # Use baseOpenInterest (coin units) × price for USD notional
    # Fall back to openInterest (already USD in some versions of the API)
    base_oi_str = market_data.get("baseOpenInterest")
    oi_str      = market_data.get("openInterest")
    if base_oi_str:
        oi_usd = float(base_oi_str) * oracle_price
    elif oi_str:
        oi_usd = float(oi_str)
    else:
        oi_usd = 0.0

    # Funding settles at top of each hour
    now = datetime.utcnow()
    next_funding = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    return MarketSnapshot(
        venue=VENUE,
        symbol=canonicalize(market),
        market=market,
        ts=now,
        mark_price=oracle_price,    # dYdX: no separate markPrice
        index_price=oracle_price,   # same → basis = 0
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
      1. Single /perpetualMarkets call → all price/funding data at once
      2. Parallel orderbook fetches for each market
    Failed markets are logged and skipped.
    """
    # Step 1: one call for all price/funding data
    try:
        all_markets = await fetch_perpetual_markets(client)
    except Exception as exc:
        log.error("dYdX /perpetualMarkets failed: %s", exc)
        return []

    async def _fetch_one(market: str) -> Optional[MarketSnapshot]:
        market_data = all_markets.get(market)
        if market_data is None:
            log.warning("dYdX: %s not found in perpetualMarkets", market)
            return None
        if market_data.get("status") != "ACTIVE":
            log.debug("dYdX: %s status=%s, skipping", market, market_data.get("status"))
            return None
        try:
            return await fetch_market_snapshot(client, market, market_data, depth)
        except Exception as exc:
            log.warning("dYdX fetch failed for %s: %s", market, exc)
            return None

    tasks = [_fetch_one(m) for m in markets]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]



async def fetch_orderbook_only(
    client: httpx.AsyncClient,
    market: str,
    depth: int = 20,
) -> Tuple[List[OrderbookLevel], List[OrderbookLevel]]:
    """Lightweight orderbook-only fetch for high-frequency polling."""
    return await fetch_orderbook(client, market, depth)



async def fetch_active_markets(client: httpx.AsyncClient) -> List[str]:
    """
    Returns list of all active dYdX market IDs (e.g. ["BTC-USD", "ETH-USD", ...]).
    """
    all_markets = await fetch_perpetual_markets(client)
    return [
        market_id
        for market_id, data in all_markets.items()
        if data.get("status") == "ACTIVE"
    ]
