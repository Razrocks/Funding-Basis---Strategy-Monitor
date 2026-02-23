"""
storage/db.py
=============
SQLite async storage layer using aiosqlite.
All schema, CRUD, and TTL cleanup live here.
The rest of the codebase only touches this module for persistence.

Swap guide: replace `aiosqlite.connect` with asyncpg/SQLAlchemy async
and adapt the SQL dialect. The interface (functions) stays the same.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiosqlite

from core.models import (
    DerivedMetrics,
    EventRecord,
    EventType,
    MarketSnapshot,
    OrderbookLevel,
    Severity,
    Venue,
)


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS market_snapshots (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    venue                    TEXT    NOT NULL,
    symbol                   TEXT    NOT NULL,
    market                   TEXT    NOT NULL,
    ts                       REAL    NOT NULL,
    mark_price               REAL    NOT NULL,
    index_price              REAL    NOT NULL,
    funding_rate             REAL    NOT NULL,
    funding_interval_seconds INTEGER NOT NULL,
    open_interest_usd        REAL    NOT NULL,
    orderbook_bids           TEXT    NOT NULL DEFAULT '[]',
    orderbook_asks           TEXT    NOT NULL DEFAULT '[]',
    next_funding_time        REAL,
    orderbook_stale          INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_snap_venue_symbol_ts
    ON market_snapshots(venue, symbol, ts DESC);

CREATE INDEX IF NOT EXISTS idx_snap_ts
    ON market_snapshots(ts DESC);

CREATE TABLE IF NOT EXISTS derived_metrics (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id              INTEGER REFERENCES market_snapshots(id) ON DELETE CASCADE,
    venue                    TEXT    NOT NULL,
    symbol                   TEXT    NOT NULL,
    market                   TEXT    NOT NULL,
    ts                       REAL    NOT NULL,
    funding_apr              REAL,
    basis                    REAL,
    basis_apr                REAL,
    gross_carry_apr          REAL,
    net_carry_apr_1k         REAL,
    net_carry_apr_5k         REAL,
    net_carry_apr_10k        REAL,
    net_carry_apr_25k        REAL,
    net_carry_apr_50k        REAL,
    net_carry_apr_100k       REAL,
    realized_funding_apr_24h REAL,
    realized_funding_apr_7d  REAL,
    funding_std_24h          REAL,
    funding_std_7d           REAL,
    basis_std_24h            REAL,
    basis_std_7d             REAL,
    carry_std_24h            REAL,
    carry_std_7d             REAL,
    oi_change_1h_pct         REAL,
    oi_change_24h_pct        REAL,
    funding_zscore_7d        REAL,
    funding_zscore_30d       REAL,
    carry_zscore_7d          REAL,
    carry_zscore_30d         REAL,
    quality_score            REAL,
    crowding_score           REAL,
    capacity_5bps            REAL,
    capacity_10bps           REAL,
    capacity_25bps           REAL,
    spread_bps               REAL,
    trap_tags                TEXT    NOT NULL DEFAULT '[]',
    carry_direction          TEXT
);

CREATE INDEX IF NOT EXISTS idx_metrics_venue_symbol_ts
    ON derived_metrics(venue, symbol, ts DESC);

CREATE INDEX IF NOT EXISTS idx_metrics_ts
    ON derived_metrics(ts DESC);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    venue       TEXT    NOT NULL,
    market      TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    details     TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_venue_ts  ON events(venue, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity  ON events(severity, ts DESC);
"""


_DB_PATH: str = "./data/monitor.db"

def set_db_path(path: str) -> None:
    global _DB_PATH
    _DB_PATH = path


@asynccontextmanager
async def get_db(path: Optional[str] = None) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Context manager that yields an open aiosqlite connection with row_factory set.

    If *path* is provided it overrides the global _DB_PATH for this connection only.
    """
    target = path if path is not None else _DB_PATH
    dir_ = os.path.dirname(target)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    async with aiosqlite.connect(target) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


async def init_db(db: Optional[aiosqlite.Connection] = None) -> None:
    """Create all tables if they don't exist.

    If an open *db* connection is provided it will be reused; otherwise a new
    connection is opened via get_db().
    """
    if db is not None:
        await db.executescript(SCHEMA_SQL)
        await db.commit()
    else:
        async with get_db() as _db:
            await _db.executescript(SCHEMA_SQL)
            await _db.commit()



def _ob_to_json(levels: List[OrderbookLevel]) -> str:
    return json.dumps([[l.price, l.size] for l in levels])

def _json_to_ob(s: str) -> List[OrderbookLevel]:
    try:
        data = json.loads(s)
        return [OrderbookLevel(price=float(p), size=float(q)) for p, q in data]
    except Exception:
        return []

def _ts(dt: datetime) -> float:
    return dt.timestamp()

def _from_ts(v: float) -> datetime:
    return datetime.utcfromtimestamp(v)



async def insert_snapshot(db: aiosqlite.Connection, snap: MarketSnapshot) -> int:
    cursor = await db.execute(
        """INSERT INTO market_snapshots
           (venue, symbol, market, ts, mark_price, index_price,
            funding_rate, funding_interval_seconds, open_interest_usd,
            orderbook_bids, orderbook_asks, next_funding_time, orderbook_stale)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            snap.venue.value, snap.symbol, snap.market, _ts(snap.ts),
            snap.mark_price, snap.index_price,
            snap.funding_rate, snap.funding_interval_seconds, snap.open_interest_usd,
            _ob_to_json(snap.orderbook_bids), _ob_to_json(snap.orderbook_asks),
            _ts(snap.next_funding_time) if snap.next_funding_time else None,
            int(snap.orderbook_stale),
        ),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_latest_snapshots(
    db: aiosqlite.Connection,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
) -> List[MarketSnapshot]:
    """Return the most recent snapshot per (venue, symbol)."""
    where_clauses = []
    params: List[Any] = []
    if venue:
        where_clauses.append("venue = ?")
        params.append(venue)
    if symbol:
        where_clauses.append("symbol = ?")
        params.append(symbol)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT * FROM market_snapshots s
        WHERE s.id = (
            SELECT id FROM market_snapshots
            WHERE venue = s.venue AND symbol = s.symbol {('AND ' + ' AND '.join(where_clauses)) if where_clauses else ''}
            ORDER BY ts DESC LIMIT 1
        ) {where}
        ORDER BY s.ts DESC
    """
    # Simpler: one row per (venue, symbol) using GROUP BY trick
    sql = f"""
        SELECT s.* FROM market_snapshots s
        INNER JOIN (
            SELECT venue, symbol, MAX(ts) as max_ts
            FROM market_snapshots
            {where}
            GROUP BY venue, symbol
        ) latest ON s.venue = latest.venue AND s.symbol = latest.symbol AND s.ts = latest.max_ts
        ORDER BY s.ts DESC
    """
    async with db.execute(sql, params) as cur:
        rows = await cur.fetchall()

    return [_row_to_snapshot(r) for r in rows]


async def get_snapshot_history(
    db: aiosqlite.Connection,
    venue: str,
    symbol: str,
    hours: int = 168,
) -> List[MarketSnapshot]:
    """Return all snapshots for a market over the last N hours."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
    sql = """SELECT * FROM market_snapshots
             WHERE venue = ? AND symbol = ? AND ts >= ?
             ORDER BY ts ASC"""
    async with db.execute(sql, (venue, symbol, cutoff)) as cur:
        rows = await cur.fetchall()
    return [_row_to_snapshot(r) for r in rows]


def _row_to_snapshot(row: aiosqlite.Row) -> MarketSnapshot:
    return MarketSnapshot(
        id=row["id"],
        venue=Venue(row["venue"]),
        symbol=row["symbol"],
        market=row["market"],
        ts=_from_ts(row["ts"]),
        mark_price=row["mark_price"],
        index_price=row["index_price"],
        funding_rate=row["funding_rate"],
        funding_interval_seconds=row["funding_interval_seconds"],
        open_interest_usd=row["open_interest_usd"],
        orderbook_bids=_json_to_ob(row["orderbook_bids"]),
        orderbook_asks=_json_to_ob(row["orderbook_asks"]),
        next_funding_time=_from_ts(row["next_funding_time"]) if row["next_funding_time"] else None,
        orderbook_stale=bool(row["orderbook_stale"]),
    )



async def insert_metrics(db: aiosqlite.Connection, m: DerivedMetrics) -> int:
    cursor = await db.execute(
        """INSERT INTO derived_metrics
           (snapshot_id, venue, symbol, market, ts,
            funding_apr, basis, basis_apr, gross_carry_apr,
            net_carry_apr_1k, net_carry_apr_5k, net_carry_apr_10k,
            net_carry_apr_25k, net_carry_apr_50k, net_carry_apr_100k,
            realized_funding_apr_24h, realized_funding_apr_7d,
            funding_std_24h, funding_std_7d,
            basis_std_24h, basis_std_7d,
            carry_std_24h, carry_std_7d,
            oi_change_1h_pct, oi_change_24h_pct,
            funding_zscore_7d, funding_zscore_30d,
            carry_zscore_7d, carry_zscore_30d,
            quality_score, crowding_score,
            capacity_5bps, capacity_10bps, capacity_25bps,
            spread_bps, trap_tags, carry_direction)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            m.snapshot_id, m.venue.value, m.symbol, m.market, _ts(m.ts),
            m.funding_apr, m.basis, m.basis_apr, m.gross_carry_apr,
            m.net_carry_apr_1k, m.net_carry_apr_5k, m.net_carry_apr_10k,
            m.net_carry_apr_25k, m.net_carry_apr_50k, m.net_carry_apr_100k,
            m.realized_funding_apr_24h, m.realized_funding_apr_7d,
            m.funding_std_24h, m.funding_std_7d,
            m.basis_std_24h, m.basis_std_7d,
            m.carry_std_24h, m.carry_std_7d,
            m.oi_change_1h_pct, m.oi_change_24h_pct,
            m.funding_zscore_7d, m.funding_zscore_30d,
            m.carry_zscore_7d, m.carry_zscore_30d,
            m.quality_score, m.crowding_score,
            m.capacity_5bps, m.capacity_10bps, m.capacity_25bps,
            m.spread_bps, json.dumps(m.trap_tags), m.carry_direction,
        ),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_latest_metrics(
    db: aiosqlite.Connection,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
) -> List[DerivedMetrics]:
    """Return most recent derived_metrics row per (venue, symbol)."""
    where_clauses, params = [], []
    if venue:
        where_clauses.append("venue = ?")
        params.append(venue)
    if symbol:
        where_clauses.append("symbol = ?")
        params.append(symbol)
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT m.* FROM derived_metrics m
        INNER JOIN (
            SELECT venue, symbol, MAX(ts) as max_ts
            FROM derived_metrics {where}
            GROUP BY venue, symbol
        ) latest ON m.venue = latest.venue AND m.symbol = latest.symbol AND m.ts = latest.max_ts
        ORDER BY m.gross_carry_apr DESC
    """
    async with db.execute(sql, params) as cur:
        rows = await cur.fetchall()
    return [_row_to_metrics(r) for r in rows]


async def get_metrics_history(
    db: aiosqlite.Connection,
    venue: str,
    symbol: str,
    hours: int = 168,
) -> List[DerivedMetrics]:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
    sql = """SELECT * FROM derived_metrics
             WHERE venue = ? AND symbol = ? AND ts >= ?
             ORDER BY ts ASC"""
    async with db.execute(sql, (venue, symbol, cutoff)) as cur:
        rows = await cur.fetchall()
    return [_row_to_metrics(r) for r in rows]


async def get_funding_rate_history(
    db: aiosqlite.Connection,
    venue: str,
    symbol: str,
    hours: int = 720,
) -> List[Tuple[float, float]]:
    """Return [(ts_unix, funding_rate), ...] for z-score / realized APR calculation."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
    sql = """SELECT ts, funding_rate FROM market_snapshots
             WHERE venue = ? AND symbol = ? AND ts >= ?
             ORDER BY ts ASC"""
    async with db.execute(sql, (venue, symbol, cutoff)) as cur:
        rows = await cur.fetchall()
    return [(r["ts"], r["funding_rate"]) for r in rows]


async def get_oi_history(
    db: aiosqlite.Connection,
    venue: str,
    symbol: str,
    hours: int = 24,
) -> List[Tuple[float, float]]:
    """Return [(ts_unix, open_interest_usd), ...] for OI change calculation."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
    sql = """SELECT ts, open_interest_usd FROM market_snapshots
             WHERE venue = ? AND symbol = ? AND ts >= ?
             ORDER BY ts ASC"""
    async with db.execute(sql, (venue, symbol, cutoff)) as cur:
        rows = await cur.fetchall()
    return [(r["ts"], r["open_interest_usd"]) for r in rows]


def _row_to_metrics(row: aiosqlite.Row) -> DerivedMetrics:
    return DerivedMetrics(
        id=row["id"],
        snapshot_id=row["snapshot_id"],
        venue=Venue(row["venue"]),
        symbol=row["symbol"],
        market=row["market"],
        ts=_from_ts(row["ts"]),
        funding_apr=row["funding_apr"],
        basis=row["basis"],
        basis_apr=row["basis_apr"],
        gross_carry_apr=row["gross_carry_apr"],
        net_carry_apr_1k=row["net_carry_apr_1k"],
        net_carry_apr_5k=row["net_carry_apr_5k"],
        net_carry_apr_10k=row["net_carry_apr_10k"],
        net_carry_apr_25k=row["net_carry_apr_25k"],
        net_carry_apr_50k=row["net_carry_apr_50k"],
        net_carry_apr_100k=row["net_carry_apr_100k"],
        realized_funding_apr_24h=row["realized_funding_apr_24h"],
        realized_funding_apr_7d=row["realized_funding_apr_7d"],
        funding_std_24h=row["funding_std_24h"],
        funding_std_7d=row["funding_std_7d"],
        basis_std_24h=row["basis_std_24h"],
        basis_std_7d=row["basis_std_7d"],
        carry_std_24h=row["carry_std_24h"],
        carry_std_7d=row["carry_std_7d"],
        oi_change_1h_pct=row["oi_change_1h_pct"],
        oi_change_24h_pct=row["oi_change_24h_pct"],
        funding_zscore_7d=row["funding_zscore_7d"],
        funding_zscore_30d=row["funding_zscore_30d"],
        carry_zscore_7d=row["carry_zscore_7d"],
        carry_zscore_30d=row["carry_zscore_30d"],
        quality_score=row["quality_score"],
        crowding_score=row["crowding_score"],
        capacity_5bps=row["capacity_5bps"],
        capacity_10bps=row["capacity_10bps"],
        capacity_25bps=row["capacity_25bps"],
        spread_bps=row["spread_bps"],
        trap_tags=json.loads(row["trap_tags"] or "[]"),
        carry_direction=row["carry_direction"],
    )



async def insert_event(db: aiosqlite.Connection, e: EventRecord) -> int:
    cursor = await db.execute(
        """INSERT INTO events (ts, venue, market, symbol, event_type, severity, message, details)
           VALUES (?,?,?,?,?,?,?,?)""",
        (_ts(e.ts), e.venue.value, e.market, e.symbol,
         e.event_type.value, e.severity.value, e.message, e.details_json()),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_recent_events(
    db: aiosqlite.Connection,
    limit: int = 200,
    severity: Optional[str] = None,
    venue: Optional[str] = None,
    hours: int = 24,
) -> List[EventRecord]:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
    where_clauses = ["ts >= ?"]
    params: List[Any] = [cutoff]
    if severity:
        where_clauses.append("severity = ?")
        params.append(severity)
    if venue:
        where_clauses.append("venue = ?")
        params.append(venue)
    where = "WHERE " + " AND ".join(where_clauses)
    sql = f"SELECT * FROM events {where} ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    async with db.execute(sql, params) as cur:
        rows = await cur.fetchall()
    return [_row_to_event(r) for r in rows]


def _row_to_event(row: aiosqlite.Row) -> EventRecord:
    return EventRecord(
        id=row["id"],
        ts=_from_ts(row["ts"]),
        venue=Venue(row["venue"]),
        market=row["market"],
        symbol=row["symbol"],
        event_type=EventType(row["event_type"]),
        severity=Severity(row["severity"]),
        message=row["message"],
        details=json.loads(row["details"] or "{}"),
    )



async def cleanup_old_data(
    db: aiosqlite.Connection,
    snapshot_ttl_hours: int = 24,
    metrics_ttl_days: int = 7,
    events_ttl_days: int = 30,
) -> Dict[str, int]:
    """Delete rows beyond TTL. Returns dict of rows deleted per table."""
    snap_cutoff    = (datetime.utcnow() - timedelta(hours=snapshot_ttl_hours)).timestamp()
    metrics_cutoff = (datetime.utcnow() - timedelta(days=metrics_ttl_days)).timestamp()
    events_cutoff  = (datetime.utcnow() - timedelta(days=events_ttl_days)).timestamp()

    c1 = await db.execute("DELETE FROM market_snapshots WHERE ts < ?", (snap_cutoff,))
    c2 = await db.execute("DELETE FROM derived_metrics   WHERE ts < ?", (metrics_cutoff,))
    c3 = await db.execute("DELETE FROM events            WHERE ts < ?", (events_cutoff,))
    await db.commit()

    return {
        "snapshots_deleted": c1.rowcount,
        "metrics_deleted":   c2.rowcount,
        "events_deleted":    c3.rowcount,
    }
