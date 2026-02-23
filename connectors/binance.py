"""
connectors/binance.py
=====================
Binance USDT-M perpetuals connector.
ONLY uses USDT-M endpoints (fapi.binance.com). No COIN-M.

Funding interval: 8 hours (28800 seconds) for all USDT-M markets.

Symbol canonicalisation: BTCUSDT → BTC  (strip trailing USDT)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx

from core.models import MarketSnapshot, OrderbookLevel, Venue

log = logging.getLogger(__name__)

BASE_URL    = "https://fapi.binance.com/fapi/v1"
VENUE       = Venue.BINANCE
INTERVAL_S  = 28800   # 8-hour funding



def canonicalize(raw: str) -> str:
    """BTCUSDT → BTC"""
    if raw.upper().endswith("USDT"):
        return raw[:-4].upper()
    if raw.upper().endswith("USD"):
        return raw[:-3].upper()
    return raw.upper()



async def fetch_premium_index(
    client: httpx.AsyncClient,
    symbol: Optional[str] = None,
) -> List[Dict]:
    """
    GET /premiumIndex
    Returns markPrice, indexPrice, lastFundingRate, nextFundingTime per symbol.
    Pass symbol=None to fetch all markets at once.
    """
    params = {}
    if symbol:
        params["symbol"] = symbol
    resp = await client.get(f"{BASE_URL}/premiumIndex", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Returns single dict when symbol given, list when not
    return data if isinstance(data, list) else [data]


async def fetch_open_interest(
    client: httpx.AsyncClient,
    symbol: str,
) -> float:
    """
    GET /openInterest?symbol=BTCUSDT
    Returns OI in base units — we multiply by mark_price for USD notional.
    """
    resp = await client.get(f"{BASE_URL}/openInterest", params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["openInterest"])


async def fetch_orderbook(
    client: httpx.AsyncClient,
    symbol: str,
    depth: int = 20,
) -> Tuple[List[OrderbookLevel], List[OrderbookLevel]]:
    """
    GET /depth?symbol=BTCUSDT&limit=20
    Valid limits: 5, 10, 20, 50, 100, 500, 1000.
    Returns (bids_desc, asks_asc).
    """
    valid_limits = [5, 10, 20, 50, 100, 500, 1000]
    limit = min(valid_limits, key=lambda x: abs(x - depth))  # snap to nearest valid
    resp = await client.get(
        f"{BASE_URL}/depth",
        params={"symbol": symbol, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    bids = [OrderbookLevel(price=float(b[0]), size=float(b[1])) for b in data["bids"]]
    asks = [OrderbookLevel(price=float(a[0]), size=float(a[1])) for a in data["asks"]]

    bids.sort(key=lambda x: x.price, reverse=True)   # best bid first
    asks.sort(key=lambda x: x.price)                  # best ask first
    return bids, asks


async def fetch_exchange_info(client: httpx.AsyncClient) -> List[str]:
    """
    GET /exchangeInfo
    Returns list of all active USDT-M market symbols.
    """
    resp = await client.get(f"{BASE_URL}/exchangeInfo", timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return [
        s["symbol"] for s in data["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
        and s["symbol"].endswith("USDT")
    ]



async def fetch_market_snapshot(
    client: httpx.AsyncClient,
    symbol: str,
    depth: int = 20,
) -> MarketSnapshot:
    """
    Fetch all data for one Binance USDT-M symbol in parallel and normalise.
    symbol: raw Binance symbol, e.g. "BTCUSDT"
    """
    # Parallel fetch: premiumIndex + OI + orderbook
    prem_task = client.get(f"{BASE_URL}/premiumIndex", params={"symbol": symbol}, timeout=10)
    oi_task   = client.get(f"{BASE_URL}/openInterest", params={"symbol": symbol}, timeout=10)
    ob_task   = client.get(f"{BASE_URL}/depth", params={"symbol": symbol, "limit": depth}, timeout=10)

    prem_resp, oi_resp, ob_resp = await asyncio.gather(prem_task, oi_task, ob_task)
    prem_resp.raise_for_status()
    oi_resp.raise_for_status()
    ob_resp.raise_for_status()

    prem = prem_resp.json()
    oi   = oi_resp.json()
    ob   = ob_resp.json()

    mark_price  = float(prem["markPrice"])
    index_price = float(prem["indexPrice"])
    fund_rate   = float(prem["lastFundingRate"])
    oi_base     = float(oi["openInterest"])
    oi_usd      = oi_base * mark_price

    next_funding = datetime.utcfromtimestamp(int(prem["nextFundingTime"]) / 1000)

    bids = [OrderbookLevel(price=float(b[0]), size=float(b[1])) for b in ob["bids"]]
    asks = [OrderbookLevel(price=float(a[0]), size=float(a[1])) for a in ob["asks"]]
    bids.sort(key=lambda x: x.price, reverse=True)
    asks.sort(key=lambda x: x.price)

    # Use Binance server timestamp for accurate data-age tracking
    _server_time = prem.get("time")
    _ts = (datetime.utcfromtimestamp(int(_server_time) / 1000)
           if _server_time else datetime.utcnow())

    return MarketSnapshot(
        venue=VENUE,
        symbol=canonicalize(symbol),
        market=symbol,
        ts=_ts,
        mark_price=mark_price,
        index_price=index_price,
        funding_rate=fund_rate,
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
    Fetch snapshots for all configured markets concurrently.
    Failed markets are logged and skipped (don't crash the whole batch).
    """
    tasks = [fetch_market_snapshot(client, sym, depth) for sym in markets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    snapshots = []
    for sym, result in zip(markets, results):
        if isinstance(result, Exception):
            log.warning("Binance fetch failed for %s: %s", sym, result)
        else:
            snapshots.append(result)

    return snapshots



async def fetch_orderbook_only(
    client: httpx.AsyncClient,
    symbol: str,
    depth: int = 20,
) -> Tuple[List[OrderbookLevel], List[OrderbookLevel]]:
    """Lightweight orderbook-only fetch for high-frequency polling."""
    return await fetch_orderbook(client, symbol, depth)



async def fetch_funding_history(
    client: httpx.AsyncClient,
    symbol: str,
    limit: int = 100,
) -> List[Dict]:
    """
    GET /fundingRate?symbol=BTCUSDT&limit=100
    Returns list of {symbol, fundingTime, fundingRate, markPrice}.
    Most recent last.
    """
    resp = await client.get(
        f"{BASE_URL}/fundingRate",
        params={"symbol": symbol, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
