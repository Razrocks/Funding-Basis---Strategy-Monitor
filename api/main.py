"""
api/main.py
===========
FastAPI backend + APScheduler background polling loop.

Start with:
  uvicorn api.main:app --reload --port 8000

Polling cadence (from config.yaml):
  - Full snapshot poll (funding + OI + orderbook): every 30s
  - Derived metrics + events computed after each poll

Routes:
  GET /api/health                              → system health + venue status
  GET /api/snapshots?venue=&symbol=            → latest snapshot per (venue, symbol)
  GET /api/metrics?venue=&symbol=              → latest derived metrics per (venue, symbol)
  GET /api/leaderboard                         → ranked leaderboard rows (sorted by gross carry)
  GET /api/events?limit=&severity=&venue=&hours= → recent events (filtered)
  GET /api/capacity/{venue}/{market}           → fresh orderbook capacity (live fetch)
  GET /api/history/{venue}/{symbol}?hours=     → time-series metrics for charts
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from connectors import binance, hyperliquid, dydx
from core.events import run_all_checks
from core.execution import build_capacity_result
from core.metrics import (
    calc_basis,
    calc_basis_apr,
    calc_carry_quality_score,
    calc_crowding_score,
    calc_funding_apr,
    calc_gross_carry_apr,
    calc_net_carry_apr,
    calc_zscore,
    carry_direction,
    get_trap_tags,
    realized_funding_apr,
    rolling_std,
)
from core.cross_venue import (
    compute_arb_leaderboard,
    compute_cross_venue_comparison,
    compute_venue_carry_leg,
)
from core.models import (
    ArbLeaderboardRow,
    CapacityResult,
    CrossVenueResult,
    DerivedMetrics,
    FillSide,
    LeaderboardRow,
    MarketSnapshot,
    Venue,
    VenueCarryLeg,
    VenueHealth,
)
from storage.db import (
    cleanup_old_data,
    get_db,
    get_funding_rate_history,
    get_latest_metrics,
    get_latest_snapshots,
    get_metrics_history,
    get_oi_history,
    get_recent_events,
    get_snapshot_history,
    init_db,
    insert_event,
    insert_metrics,
    insert_snapshot,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

CFG = load_config()


_venue_health: Dict[str, dict] = {
    "binance":     {"last_poll_ts": None, "last_latency_ms": None, "consecutive_errors": 0, "errors_1h": []},
    "hyperliquid": {"last_poll_ts": None, "last_latency_ms": None, "consecutive_errors": 0, "errors_1h": []},
    "dydx":        {"last_poll_ts": None, "last_latency_ms": None, "consecutive_errors": 0, "errors_1h": []},
}

def _record_success(venue_key: str, latency_ms: float) -> None:
    h = _venue_health[venue_key]
    h["last_poll_ts"]       = datetime.utcnow()
    h["last_latency_ms"]    = latency_ms
    h["consecutive_errors"] = 0
    # prune old 1h errors
    cutoff = time.time() - 3600
    h["errors_1h"] = [t for t in h["errors_1h"] if t > cutoff]

def _record_error(venue_key: str) -> None:
    h = _venue_health[venue_key]
    h["consecutive_errors"] += 1
    h["errors_1h"].append(time.time())

def _venue_status(venue_key: str) -> str:
    h = _venue_health[venue_key]
    if h["consecutive_errors"] >= 5:
        return "down"
    if h["consecutive_errors"] >= 2:
        return "degraded"
    return "ok"



async def _compute_and_store_metrics(
    snap: MarketSnapshot,
    snap_id: int,
    db,
    cfg: dict,
) -> DerivedMetrics:
    """
    Given a fresh snapshot, pull historical data from DB, compute all metrics,
    insert into DB, and return the DerivedMetrics object.
    """
    venue_key    = snap.venue.value
    venue_cfg    = cfg["venues"][venue_key]
    taker_fee    = venue_cfg["taker_fee_bps"]
    hold_days    = cfg["metrics"]["assumed_hold_days"]
    basis_horizon = cfg["metrics"]["basis_horizon_seconds"]
    borrow_apr   = cfg["borrow_rates"]["default_apr"]
    size_grid    = cfg["execution"]["size_grid_usd"]

    funding_apr  = calc_funding_apr(snap.funding_rate, snap.funding_interval_seconds)
    basis        = calc_basis(snap.mark_price, snap.index_price)
    basis_apr_v  = calc_basis_apr(basis, basis_horizon)
    gross_carry  = calc_gross_carry_apr(funding_apr, basis_apr_v)
    direction    = carry_direction(snap.funding_rate)

    fund_history_24h = await get_funding_rate_history(db, snap.venue, snap.symbol, hours=24)
    fund_history_7d  = await get_funding_rate_history(db, snap.venue, snap.symbol, hours=168)
    fund_history_30d = await get_funding_rate_history(db, snap.venue, snap.symbol, hours=720)
    oi_history_1h    = await get_oi_history(db, snap.venue, snap.symbol, hours=1)
    oi_history_24h   = await get_oi_history(db, snap.venue, snap.symbol, hours=24)

    realized_24h = realized_funding_apr([r[1] for r in fund_history_24h], snap.funding_interval_seconds) if fund_history_24h else None
    realized_7d  = realized_funding_apr([r[1] for r in fund_history_7d],  snap.funding_interval_seconds) if fund_history_7d  else None

    f_aprs_24h = [calc_funding_apr(r[1], snap.funding_interval_seconds) for r in fund_history_24h]
    f_aprs_7d  = [calc_funding_apr(r[1], snap.funding_interval_seconds) for r in fund_history_7d]

    funding_std_24h = rolling_std(f_aprs_24h)
    funding_std_7d  = rolling_std(f_aprs_7d)

    # carry std = std of (funding_apr) since basis is very slowly changing
    carry_std_24h = funding_std_24h
    carry_std_7d  = funding_std_7d

    f_aprs_30d = [calc_funding_apr(r[1], snap.funding_interval_seconds) for r in fund_history_30d]
    funding_zscore_7d  = calc_zscore(funding_apr, f_aprs_7d)  if len(f_aprs_7d)  >= 10 else None
    funding_zscore_30d = calc_zscore(funding_apr, f_aprs_30d) if len(f_aprs_30d) >= 10 else None
    carry_zscore_7d    = funding_zscore_7d   # carry = funding for most markets
    carry_zscore_30d   = funding_zscore_30d

    oi_change_1h_pct  = None
    oi_change_24h_pct = None
    if oi_history_1h:
        prev_oi = oi_history_1h[0][1]   # (ts, oi_usd) → take oi_usd
        if prev_oi and prev_oi > 0:
            oi_change_1h_pct = (snap.open_interest_usd - prev_oi) / prev_oi * 100
    if oi_history_24h:
        prev_oi_24 = oi_history_24h[0][1]
        if prev_oi_24 and prev_oi_24 > 0:
            oi_change_24h_pct = (snap.open_interest_usd - prev_oi_24) / prev_oi_24 * 100

    quality_score  = calc_carry_quality_score(
        gross_carry_apr=gross_carry,
        funding_std_24h=funding_std_24h,
        basis_std_24h=None,   # not tracking basis history per snapshot currently
        carry_std_24h=carry_std_24h,
        funding_zscore=funding_zscore_30d or funding_zscore_7d,
        oi_change_1h_pct=oi_change_1h_pct,
    )
    crowding_score = calc_crowding_score(
        funding_zscore=funding_zscore_30d or funding_zscore_7d,
        oi_change_24h_pct=oi_change_24h_pct,
        carry_std_7d=carry_std_7d,
        gross_carry_apr=gross_carry,
    )
    trap_tags = get_trap_tags(
        funding_zscore=funding_zscore_30d or funding_zscore_7d,
        oi_change_1h_pct=oi_change_1h_pct,
        carry_std_24h=carry_std_24h,
        gross_carry_apr=gross_carry,
        basis=basis,
    )

    cap_result = build_capacity_result(
        venue=snap.venue,
        symbol=snap.symbol,
        market=snap.market,
        bids=snap.orderbook_bids,
        asks=snap.orderbook_asks,
        size_grid=size_grid,
    )

    # Net carry at each size in the grid
    fill_side = FillSide.SELL if snap.funding_rate > 0 else FillSide.BUY
    curve = cap_result.slippage_curve_sell if fill_side == FillSide.SELL else cap_result.slippage_curve_buy

    def _net_at_idx(idx: int) -> Optional[float]:
        if idx >= len(curve):
            return None
        sp = curve[idx]
        if sp.slippage_bps >= 9000:
            return None
        return calc_net_carry_apr(
            funding_apr=funding_apr,
            basis_apr=basis_apr_v,
            slippage_bps=sp.slippage_bps,
            taker_fee_bps=taker_fee,
            borrow_apr=borrow_apr,
            hold_days=hold_days,
        )

    net_at_sizes = {
        "1k":   _net_at_idx(0),
        "5k":   _net_at_idx(1),
        "10k":  _net_at_idx(2),
        "25k":  _net_at_idx(3),
        "50k":  _net_at_idx(4),
        "100k": _net_at_idx(5),
    }

    metrics = DerivedMetrics(
        snapshot_id=snap_id,
        venue=snap.venue,
        symbol=snap.symbol,
        market=snap.market,
        ts=snap.ts,
        funding_apr=funding_apr,
        basis=basis,
        basis_apr=basis_apr_v,
        gross_carry_apr=gross_carry,
        net_carry_apr_1k=net_at_sizes["1k"],
        net_carry_apr_5k=net_at_sizes["5k"],
        net_carry_apr_10k=net_at_sizes["10k"],
        net_carry_apr_25k=net_at_sizes["25k"],
        net_carry_apr_50k=net_at_sizes["50k"],
        net_carry_apr_100k=net_at_sizes["100k"],
        realized_funding_apr_24h=realized_24h,
        realized_funding_apr_7d=realized_7d,
        funding_std_24h=funding_std_24h,
        funding_std_7d=funding_std_7d,
        carry_std_24h=carry_std_24h,
        carry_std_7d=carry_std_7d,
        oi_change_1h_pct=oi_change_1h_pct,
        oi_change_24h_pct=oi_change_24h_pct,
        funding_zscore_7d=funding_zscore_7d,
        funding_zscore_30d=funding_zscore_30d,
        carry_zscore_7d=carry_zscore_7d,
        carry_zscore_30d=carry_zscore_30d,
        quality_score=quality_score,
        crowding_score=crowding_score,
        capacity_5bps=cap_result.capacity_5bps_min,
        capacity_10bps=cap_result.capacity_10bps_min,
        capacity_25bps=cap_result.capacity_25bps_min,
        spread_bps=cap_result.spread_bps,
        trap_tags=trap_tags,
        carry_direction=direction,
    )

    metrics_id = await insert_metrics(db, metrics)
    metrics.id = metrics_id

    # Get previous snapshot for prev_funding_rate + prev OI
    prev_snap_history = await get_snapshot_history(db, snap.venue, snap.symbol, hours=2)
    prev_funding_rate = prev_snap_history[-2].funding_rate if len(prev_snap_history) >= 2 else None
    prev_oi           = oi_history_1h[0][1] if oi_history_1h else None   # (ts, oi_usd) → oi_usd
    prev_carry_std    = carry_std_24h  # same window for now

    events = run_all_checks(
        venue=snap.venue,
        market=snap.market,
        symbol=snap.symbol,
        funding_apr=funding_apr,
        basis=basis,
        gross_carry_apr=gross_carry,
        net_carry_apr=net_at_sizes["25k"],
        quality_score=quality_score,
        capacity_10bps=cap_result.capacity_10bps_min,
        oi_now=snap.open_interest_usd,
        oi_prev=prev_oi,
        carry_std_24h=carry_std_24h,
        carry_std_prev=prev_carry_std,
        prev_funding_rate=prev_funding_rate,
        curr_funding_rate=snap.funding_rate,
        funding_spike_threshold=cfg["events"]["funding_spike_threshold"],
        basis_inversion_threshold=cfg["events"]["basis_inversion_threshold"],
        oi_shock_threshold_pct=cfg["events"]["oi_change_1h_threshold_pct"],
        carry_vol_cv_threshold=cfg["events"]["carry_vol_threshold_cv"],
    )

    for evt in events:
        await insert_event(db, evt)
        log.info("EVENT [%s] %s", evt.severity.value.upper(), evt.message)

    return metrics



async def _poll_binance(cfg: dict) -> None:
    venue_cfg = cfg["venues"]["binance"]
    if not venue_cfg.get("enabled", True):
        return

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient() as client:
            snapshots = await binance.fetch_all_snapshots(
                client,
                markets=venue_cfg["markets"],
                depth=cfg["orderbook_depth"],
            )
        latency_ms = (time.monotonic() - t0) * 1000
        _record_success("binance", latency_ms)

        async with get_db(cfg["storage"]["db_path"]) as db:
            for snap in snapshots:
                snap_id = await insert_snapshot(db, snap)
                await _compute_and_store_metrics(snap, snap_id, db, cfg)

        log.info("Binance: polled %d markets in %.0fms", len(snapshots), latency_ms)

    except Exception as exc:
        _record_error("binance")
        log.error("Binance poll error: %s", exc, exc_info=True)


async def _poll_hyperliquid(cfg: dict) -> None:
    venue_cfg = cfg["venues"]["hyperliquid"]
    if not venue_cfg.get("enabled", True):
        return

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient() as client:
            snapshots = await hyperliquid.fetch_all_snapshots(
                client,
                markets=venue_cfg["markets"],
                depth=cfg["orderbook_depth"],
            )
        latency_ms = (time.monotonic() - t0) * 1000
        _record_success("hyperliquid", latency_ms)

        async with get_db(cfg["storage"]["db_path"]) as db:
            for snap in snapshots:
                snap_id = await insert_snapshot(db, snap)
                await _compute_and_store_metrics(snap, snap_id, db, cfg)

        log.info("Hyperliquid: polled %d markets in %.0fms", len(snapshots), latency_ms)

    except Exception as exc:
        _record_error("hyperliquid")
        log.error("Hyperliquid poll error: %s", exc, exc_info=True)


async def _poll_dydx(cfg: dict) -> None:
    venue_cfg = cfg["venues"]["dydx"]
    if not venue_cfg.get("enabled", True):
        return

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient() as client:
            snapshots = await dydx.fetch_all_snapshots(
                client,
                markets=venue_cfg["markets"],
                depth=cfg["orderbook_depth"],
            )
        latency_ms = (time.monotonic() - t0) * 1000
        _record_success("dydx", latency_ms)

        async with get_db(cfg["storage"]["db_path"]) as db:
            for snap in snapshots:
                snap_id = await insert_snapshot(db, snap)
                await _compute_and_store_metrics(snap, snap_id, db, cfg)

        log.info("dYdX: polled %d markets in %.0fms", len(snapshots), latency_ms)

    except Exception as exc:
        _record_error("dydx")
        log.error("dYdX poll error: %s", exc, exc_info=True)


async def poll_all_venues() -> None:
    """Run all venue polls concurrently."""
    cfg = load_config()
    await asyncio.gather(
        _poll_binance(cfg),
        _poll_hyperliquid(cfg),
        _poll_dydx(cfg),
    )

    # Periodic cleanup (runs every poll — DB handles TTL efficiently via index)
    try:
        async with get_db(cfg["storage"]["db_path"]) as db:
            await cleanup_old_data(
                db,
                snapshot_ttl_hours=cfg["storage"]["full_resolution_ttl_hours"],
                metrics_ttl_days=cfg["storage"]["derived_metrics_ttl_days"],
                events_ttl_days=cfg["storage"]["events_ttl_days"],
            )
    except Exception as exc:
        log.warning("DB cleanup error: %s", exc)



scheduler = AsyncIOScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()

    # Ensure data directory exists
    db_path = Path(cfg["storage"]["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Init DB schema
    async with get_db(str(db_path)) as db:
        await init_db(db)
    log.info("Database initialised at %s", db_path)

    # Run first poll immediately (so API has data within seconds of startup)
    log.info("Running initial poll on startup …")
    await poll_all_venues()

    # Schedule recurring polls
    interval = cfg["polling_interval_seconds"]
    scheduler.add_job(poll_all_venues, "interval", seconds=interval, id="poll_all")
    scheduler.start()
    log.info("Scheduler started: polling every %ds", interval)

    yield

    scheduler.shutdown(wait=False)
    log.info("Scheduler stopped")


app = FastAPI(
    title="Crypto Funding & Basis Carry Monitor",
    version="1.0.0",
    lifespan=lifespan,
)

cfg_cors = CFG.get("api", {}).get("cors_origins", ["http://localhost:8501"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg_cors,
    allow_methods=["GET"],
    allow_headers=["*"],
)



@app.get("/api/health")
async def health():
    """System health + per-venue status."""
    cfg = load_config()
    venues_out = {}
    for venue_key, h in _venue_health.items():
        cutoff = time.time() - 3600
        errors_1h = len([t for t in h["errors_1h"] if t > cutoff])
        venues_out[venue_key] = VenueHealth(
            venue=Venue(venue_key),
            last_poll_ts=h["last_poll_ts"],
            last_poll_latency_ms=h["last_latency_ms"],
            consecutive_errors=h["consecutive_errors"],
            total_errors_1h=errors_1h,
            markets_active=len(cfg["venues"][venue_key].get("markets", [])),
            status=_venue_status(venue_key),
        ).model_dump()
    return {
        "status": "ok",
        "ts": datetime.utcnow().isoformat(),
        "venues": venues_out,
    }


@app.get("/api/snapshots")
async def get_snapshots(
    venue: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
):
    """Latest raw market snapshot per (venue, symbol) pair."""
    cfg = load_config()
    venue_enum = Venue(venue) if venue else None
    async with get_db(cfg["storage"]["db_path"]) as db:
        rows = await get_latest_snapshots(db, venue=venue_enum, symbol=symbol)
    return {"snapshots": [r.model_dump() for r in rows], "count": len(rows)}


@app.get("/api/metrics")
async def get_metrics(
    venue: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
):
    """Latest derived metrics per (venue, symbol) pair."""
    cfg = load_config()
    venue_enum = Venue(venue) if venue else None
    async with get_db(cfg["storage"]["db_path"]) as db:
        rows = await get_latest_metrics(db, venue=venue_enum, symbol=symbol)
    return {"metrics": [r.model_dump() for r in rows], "count": len(rows)}


@app.get("/api/leaderboard")
async def get_leaderboard(
    min_quality: float = Query(0.0),
    min_gross_carry: float = Query(0.0),
    venue: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    sort_by: str = Query("gross_carry_apr"),
    limit: int = Query(50, le=200),
):
    """
    Ranked leaderboard combining latest metrics + snapshot data.
    Sorted by gross_carry_apr (descending) by default.
    """
    cfg = load_config()
    venue_enum = Venue(venue) if venue else None

    async with get_db(cfg["storage"]["db_path"]) as db:
        all_metrics = await get_latest_metrics(db, venue=venue_enum, symbol=symbol)
        all_snaps   = await get_latest_snapshots(db, venue=venue_enum, symbol=symbol)

    # Build a (venue, symbol) → snapshot lookup
    snap_lookup = {(s.venue, s.symbol): s for s in all_snaps}

    rows: List[LeaderboardRow] = []
    for m in all_metrics:
        if (m.quality_score or 0) < min_quality:
            continue
        if m.gross_carry_apr < min_gross_carry:
            continue
        snap = snap_lookup.get((m.venue, m.symbol))
        rows.append(LeaderboardRow(
            rank=0,           # assigned after sort
            venue=m.venue.value,
            symbol=m.symbol,
            market=m.market,
            ts=m.ts,
            mark_price=snap.mark_price if snap else 0.0,
            funding_apr=m.funding_apr,
            basis_apr=m.basis_apr,
            gross_carry_apr=m.gross_carry_apr,
            net_carry_apr_25k=m.net_carry_apr_25k,
            capacity_5bps=m.capacity_5bps,
            capacity_10bps=m.capacity_10bps,
            capacity_25bps=m.capacity_25bps,
            open_interest_usd=snap.open_interest_usd if snap else 0.0,
            quality_score=m.quality_score,
            crowding_score=m.crowding_score,
            spread_bps=m.spread_bps,
            trap_tags=m.trap_tags,
            carry_direction=m.carry_direction,
            next_funding_time=snap.next_funding_time if snap else None,
            funding_interval_seconds=snap.funding_interval_seconds if snap else 3600,
        ))

    # Sort
    reverse = True
    if sort_by == "gross_carry_apr":
        rows.sort(key=lambda r: r.gross_carry_apr, reverse=reverse)
    elif sort_by == "net_carry_apr_25k":
        rows.sort(key=lambda r: r.net_carry_apr_25k or 0, reverse=reverse)
    elif sort_by == "quality_score":
        rows.sort(key=lambda r: r.quality_score or 0, reverse=reverse)
    elif sort_by == "capacity_10bps":
        rows.sort(key=lambda r: r.capacity_10bps or 0, reverse=reverse)
    elif sort_by == "open_interest_usd":
        rows.sort(key=lambda r: r.open_interest_usd, reverse=reverse)

    # Assign ranks and limit
    rows = rows[:limit]
    for i, row in enumerate(rows, 1):
        row.rank = i

    return {"rows": [r.model_dump() for r in rows], "count": len(rows)}


@app.get("/api/events")
async def get_events(
    limit: int = Query(100, le=500),
    severity: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    hours: float = Query(24.0),
):
    """Recent events (filtered). Hours window: 1, 6, 24, 168 (7d)."""
    cfg = load_config()
    venue_enum = Venue(venue) if venue else None
    async with get_db(cfg["storage"]["db_path"]) as db:
        events = await get_recent_events(
            db,
            limit=limit,
            severity=severity,
            venue=venue_enum,
            hours=hours,
        )
    return {"events": [e.model_dump() for e in events], "count": len(events)}


@app.get("/api/capacity/{venue}/{market}")
async def get_live_capacity(venue: str, market: str):
    """
    Fetch a live orderbook and return capacity + slippage curve.
    Always fetches fresh (not cached) — use sparingly.
    """
    cfg = load_config()
    depth = cfg["orderbook_depth"]

    try:
        async with httpx.AsyncClient() as client:
            if venue == "binance":
                bids, asks = await binance.fetch_orderbook(client, market, depth)
            elif venue == "hyperliquid":
                bids, asks = await hyperliquid.fetch_orderbook(client, market, depth)
            elif venue == "dydx":
                bids, asks = await dydx.fetch_orderbook(client, market, depth)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown venue: {venue}")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    from core.models import Venue as VenueEnum
    cap = build_capacity_result(
        venue=VenueEnum(venue),
        symbol=market.split("-")[0].replace("USDT", "").upper(),
        market=market,
        bids=bids,
        asks=asks,
        size_grid=cfg["execution"]["size_grid_usd"],
    )
    return cap.model_dump()


@app.get("/api/history/{venue}/{symbol}")
async def get_history(
    venue: str,
    symbol: str,
    hours: float = Query(24.0),
):
    """
    Time-series metrics history for charts.
    Returns both snapshot fields and derived metrics over the window.
    """
    cfg = load_config()
    try:
        venue_enum = Venue(venue)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown venue: {venue}")

    async with get_db(cfg["storage"]["db_path"]) as db:
        metrics_hist = await get_metrics_history(db, venue_enum, symbol, hours=hours)
        snap_hist    = await get_snapshot_history(db, venue_enum, symbol, hours=hours)

    return {
        "venue": venue,
        "symbol": symbol,
        "hours": hours,
        "metrics": [m.model_dump() for m in metrics_hist],
        "snapshots": [s.model_dump() for s in snap_hist],
        "count": len(metrics_hist),
    }


@app.get("/api/orderbook/{venue}/{market}")
async def get_live_orderbook(venue: str, market: str, depth: int = Query(20, le=100)):
    """
    Live orderbook levels (fresh fetch).
    Returns bids + asks + spread + top-of-book prices.
    """
    cfg = load_config()

    try:
        async with httpx.AsyncClient() as client:
            if venue == "binance":
                bids, asks = await binance.fetch_orderbook(client, market, depth)
            elif venue == "hyperliquid":
                bids, asks = await hyperliquid.fetch_orderbook(client, market, depth)
            elif venue == "dydx":
                bids, asks = await dydx.fetch_orderbook(client, market, depth)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown venue: {venue}")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    best_bid = bids[0].price if bids else None
    best_ask = asks[0].price if asks else None
    spread = None
    if best_bid and best_ask and best_bid > 0:
        spread = (best_ask - best_bid) / best_bid * 10_000

    return {
        "venue": venue,
        "market": market,
        "ts": datetime.utcnow().isoformat(),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_bps": spread,
        "bids": [{"price": b.price, "size": b.size} for b in bids],
        "asks": [{"price": a.price, "size": a.size} for a in asks],
    }



def _venue_fee_bps(cfg: dict, venue_key: str) -> float:
    """Return taker fee in bps for a venue from config."""
    return cfg["venues"].get(venue_key, {}).get("taker_fee_bps", 4.0)


def _build_cross_venue_result(
    symbol: str,
    all_snaps: List[MarketSnapshot],
    all_metrics: List[DerivedMetrics],
    cfg: dict,
    size_usd: float,
    hold_days: int,
    funding_only: bool = False,
) -> CrossVenueResult:
    """Build a CrossVenueResult for one symbol from latest DB data."""
    snap_by_venue = {s.venue.value: s for s in all_snaps if s.symbol == symbol}
    met_by_venue = {m.venue.value: m for m in all_metrics if m.symbol == symbol}

    legs: List[VenueCarryLeg] = []
    for venue_key, snap in snap_by_venue.items():
        fee = _venue_fee_bps(cfg, venue_key)
        metrics = met_by_venue.get(venue_key)
        leg = compute_venue_carry_leg(
            snap, metrics, size_usd, fee, hold_days, funding_only=funding_only,
        )
        legs.append(leg)

    return compute_cross_venue_comparison(symbol, legs, size_usd, hold_days)


@app.get("/api/cross-venue/{symbol}")
async def get_cross_venue(
    symbol: str,
    size: float = Query(25000),
    hold_days: int = Query(30),
    funding_only: bool = Query(False),
):
    """
    Cross-venue carry comparison for a single asset.
    Returns per-venue breakdown + optimal earn/hedge pair + edge.
    """
    cfg = load_config()
    async with get_db(cfg["storage"]["db_path"]) as db:
        all_snaps = await get_latest_snapshots(db)
        all_metrics = await get_latest_metrics(db)

    result = _build_cross_venue_result(
        symbol.upper(), all_snaps, all_metrics, cfg, size, hold_days,
        funding_only=funding_only,
    )

    if not result.venues:
        raise HTTPException(status_code=404, detail=f"No data for symbol: {symbol}")

    return result.model_dump()


@app.get("/api/arb-leaderboard")
async def get_arb_leaderboard(
    size: float = Query(25000),
    min_edge: float = Query(0.0),
    hold_days: int = Query(30),
    limit: int = Query(50, le=200),
    funding_only: bool = Query(False),
):
    """
    Cross-venue arb leaderboard ranked by edge_apr descending.
    Only includes symbols available on 2+ venues.
    """
    cfg = load_config()
    async with get_db(cfg["storage"]["db_path"]) as db:
        all_snaps = await get_latest_snapshots(db)
        all_metrics = await get_latest_metrics(db)

    # Find all symbols that appear on 2+ venues
    from collections import Counter
    symbol_venues = Counter(s.symbol for s in all_snaps)
    multi_venue_symbols = [sym for sym, cnt in symbol_venues.items() if cnt >= 2]

    # Build cross-venue result per symbol
    results: List[CrossVenueResult] = []
    for sym in multi_venue_symbols:
        cvr = _build_cross_venue_result(
            sym, all_snaps, all_metrics, cfg, size, hold_days,
            funding_only=funding_only,
        )
        results.append(cvr)

    rows = compute_arb_leaderboard(results, min_edge=min_edge)
    rows = rows[:limit]

    return {"rows": [r.model_dump() for r in rows], "count": len(rows)}
