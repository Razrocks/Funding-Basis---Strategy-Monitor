"""
core/models.py
==============
Canonical normalized data models shared across all venues, metrics, storage, and API.

Convention (explicit):
  - funding_rate > 0  → longs pay shorts → SHORT the perp earns funding
  - funding_rate < 0  → shorts pay longs → LONG the perp earns funding
  - All notional sizes are in USDT.
  - open_interest is stored in USD notional (base_qty × mark_price at fetch time).
  - funding_rate is the raw per-interval rate (not annualised).
  - funding_interval_seconds: Binance=28800 (8h), Hyperliquid=3600 (1h), dYdX=3600 (1h).
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

class Venue(str, Enum):
    BINANCE     = "binance"
    HYPERLIQUID = "hyperliquid"
    DYDX        = "dydx"


class Severity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


class EventType(str, Enum):
    FUNDING_SPIKE      = "funding_spike"
    FUNDING_FLIP       = "funding_flip"
    BASIS_INVERSION    = "basis_inversion"
    OI_SHOCK           = "oi_shock"
    CARRY_UNSTABLE     = "carry_unstable"
    REGIME_SHIFT       = "regime_shift"
    CARRY_OPPORTUNITY  = "carry_opportunity"
    VENUE_ERROR        = "venue_error"


class FillSide(str, Enum):
    BUY  = "buy"   # consume asks — long perp earns when funding < 0
    SELL = "sell"  # consume bids — short perp earns when funding > 0

class OrderbookLevel(BaseModel):
    """Single price level in an orderbook."""
    price: float
    size: float  # base asset quantity

    @property
    def notional(self) -> float:
        return self.price * self.size

class MarketSnapshot(BaseModel):
    """
    Normalized snapshot of a single perp market at a point in time.
    Produced by venue connectors; stored in market_snapshots table.
    """
    id: Optional[int] = None

    # Identity
    venue: Venue
    symbol: str   # canonical: "BTC", "ETH", "SOL"
    market: str   # raw venue market id: "BTCUSDT", "BTC", "BTC-USD"
    ts: datetime

    # Prices
    mark_price: float
    index_price: float   # oracle / spot proxy; equals mark_price for dYdX (no separate mark)

    # Funding
    funding_rate: float             # raw per-interval rate (e.g. 0.0001 for 0.01%)
    funding_interval_seconds: int   # seconds per funding period

    # Open interest (USD notional)
    open_interest_usd: float

    # Orderbook (top N levels, sorted: bids descending, asks ascending)
    orderbook_bids: List[OrderbookLevel] = Field(default_factory=list)
    orderbook_asks: List[OrderbookLevel] = Field(default_factory=list)

    # Next funding timestamp (if available from API)
    next_funding_time: Optional[datetime] = None

    # Data quality flag
    orderbook_stale: bool = False   # True if orderbook could not be fetched this cycle

    @property
    def best_bid(self) -> Optional[float]:
        return self.orderbook_bids[0].price if self.orderbook_bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.orderbook_asks[0].price if self.orderbook_asks else None

    @property
    def spread_bps(self) -> Optional[float]:
        if self.best_bid and self.best_ask and self.best_bid > 0:
            return (self.best_ask - self.best_bid) / self.best_bid * 10_000
        return None

    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.mark_price

    @property
    def seconds_to_next_funding(self) -> Optional[int]:
        if self.next_funding_time is None:
            return None
        delta = (self.next_funding_time - datetime.utcnow()).total_seconds()
        return max(0, int(delta))

class DerivedMetrics(BaseModel):
    """
    Computed metrics derived from MarketSnapshot(s).
    Stored in derived_metrics table alongside the raw snapshot.
    """
    id: Optional[int] = None
    snapshot_id: Optional[int] = None

    venue: Venue
    symbol: str
    market: str
    ts: datetime

    # Core carry components
    funding_apr: float          # annualised funding rate
    basis: float                # (mark/index) - 1; always 0.0 for dYdX
    basis_apr: float            # annualised basis (basis * 365d / horizon)
    gross_carry_apr: float      # funding_apr + basis_apr

    # Net carry at specific sizes (after fees + slippage, amortised over 30d hold)
    net_carry_apr_1k:   Optional[float] = None
    net_carry_apr_5k:   Optional[float] = None
    net_carry_apr_10k:  Optional[float] = None
    net_carry_apr_25k:  Optional[float] = None
    net_carry_apr_50k:  Optional[float] = None
    net_carry_apr_100k: Optional[float] = None

    # Realized carry from history (requires stored snapshots)
    realized_funding_apr_24h: Optional[float] = None
    realized_funding_apr_7d:  Optional[float] = None

    # Rolling volatility
    funding_std_24h: Optional[float] = None   # std dev of funding_apr over last 24h
    funding_std_7d:  Optional[float] = None
    basis_std_24h:   Optional[float] = None
    basis_std_7d:    Optional[float] = None
    carry_std_24h:   Optional[float] = None   # std dev of (funding_apr + basis_apr)
    carry_std_7d:    Optional[float] = None

    # OI change
    oi_change_1h_pct:  Optional[float] = None
    oi_change_24h_pct: Optional[float] = None

    # Z-scores
    funding_zscore_7d:  Optional[float] = None
    funding_zscore_30d: Optional[float] = None
    carry_zscore_7d:    Optional[float] = None
    carry_zscore_30d:   Optional[float] = None

    # Composite scores
    quality_score:   Optional[float] = None   # 0-100 carry quality
    crowding_score:  Optional[float] = None   # 0-100 crowding proxy

    # Capacity (from orderbook)
    capacity_5bps:  Optional[float] = None    # USD notional at ≤5 bps slippage
    capacity_10bps: Optional[float] = None
    capacity_25bps: Optional[float] = None
    spread_bps:     Optional[float] = None

    # Trap tags (string list, serialised as JSON in DB)
    trap_tags: List[str] = Field(default_factory=list)

    # Carry direction
    carry_direction: Optional[str] = None   # "SHORT" or "LONG"

class SlippagePoint(BaseModel):
    """Result of simulating a single fill at a given notional size."""
    size_usd: float
    avg_fill_price: float
    executed_notional: float   # may be < size_usd if book runs out or slippage breached
    unfilled_notional: float   # size_usd - executed_notional
    slippage_bps: float        # 9999 if fill impossible
    levels_used: int
    fill_complete: bool        # True if executed_notional >= size_usd * 0.999

    @property
    def fill_pct(self) -> float:
        if self.size_usd <= 0:
            return 0.0
        return min(100.0, self.executed_notional / self.size_usd * 100)


class CapacityResult(BaseModel):
    """Orderbook capacity and slippage curve for a market at a point in time."""
    venue: Venue
    symbol: str
    market: str
    ts: datetime

    # Capacity: max USD notional fillable within slippage threshold (sell side / bid side)
    capacity_5bps:  float
    capacity_10bps: float
    capacity_25bps: float

    # Ask-side capacity (for long entries)
    capacity_5bps_buy:  float
    capacity_10bps_buy: float
    capacity_25bps_buy: float

    # Min of buy/sell capacity (binding constraint)
    @property
    def capacity_5bps_min(self) -> float:
        return min(self.capacity_5bps, self.capacity_5bps_buy)

    @property
    def capacity_10bps_min(self) -> float:
        return min(self.capacity_10bps, self.capacity_10bps_buy)

    @property
    def capacity_25bps_min(self) -> float:
        return min(self.capacity_25bps, self.capacity_25bps_buy)

    # Full slippage curve across size grid
    slippage_curve_sell: List[SlippagePoint] = Field(default_factory=list)
    slippage_curve_buy:  List[SlippagePoint] = Field(default_factory=list)

    # Spread
    spread_bps: Optional[float] = None

    # Number of levels available
    bid_levels: int = 0
    ask_levels: int = 0

class EventRecord(BaseModel):
    """
    An anomaly or signal detected during a polling cycle.
    Stored in the events table.
    """
    id: Optional[int] = None
    ts: datetime

    venue: Venue
    market: str   # raw venue market id (e.g. "BTCUSDT")
    symbol: str   # canonical symbol (e.g. "BTC")

    event_type: EventType
    severity: Severity

    # Human-readable description
    message: str

    # Structured details for programmatic consumption
    details: Dict[str, Any] = Field(default_factory=dict)

    def details_json(self) -> str:
        return json.dumps(self.details)

class LeaderboardRow(BaseModel):
    """Flattened row for the leaderboard API endpoint."""
    rank: int
    venue: str
    symbol: str
    market: str
    ts: datetime
    mark_price: float
    funding_apr: float
    basis_apr: float
    gross_carry_apr: float
    net_carry_apr_25k: Optional[float]
    capacity_5bps: Optional[float]
    capacity_10bps: Optional[float]
    capacity_25bps: Optional[float]
    open_interest_usd: float
    quality_score: Optional[float]
    crowding_score: Optional[float]
    spread_bps: Optional[float]
    trap_tags: List[str]
    carry_direction: Optional[str]
    next_funding_time: Optional[datetime]
    funding_interval_seconds: int


class VenueHealth(BaseModel):
    """Status of a venue connector."""
    venue: Venue
    last_poll_ts: Optional[datetime]
    last_poll_latency_ms: Optional[float]
    consecutive_errors: int
    total_errors_1h: int
    markets_active: int
    status: str   # "ok", "degraded", "down"

class VenueCarryLeg(BaseModel):
    """Single-venue carry breakdown at a given trade size."""
    venue: str
    market: str
    funding_rate: float               # raw per-interval rate
    funding_interval_seconds: int
    funding_apr: float
    basis_apr: Optional[float]        # None for dYdX (oracle=mark)
    gross_carry_apr: float            # funding_apr + basis_apr (or funding-only for dYdX)
    slippage_bps: float               # actual slippage at the requested size
    taker_fee_bps: float
    cost_apr: float                   # annualised total round-trip cost
    fee_cost_apr: float               # annualised round-trip taker fee cost
    slip_cost_apr: float              # annualised round-trip slippage cost
    borrow_apr: float = 0.0           # annualised borrow rate
    net_carry_apr: float              # gross - cost
    carry_direction: str              # "SHORT" or "LONG"
    capacity_sell_10bps: float        # USD fillable selling within 10bps
    capacity_buy_10bps: float         # USD fillable buying within 10bps
    spread_bps: Optional[float]
    quality_score: Optional[float]
    trap_tags: List[str] = Field(default_factory=list)
    mark_price: float
    index_price: float
    open_interest_usd: float
    next_funding_time: Optional[datetime]
    data_ts: datetime
    data_age_seconds: float


class CrossVenueArbPair(BaseModel):
    """Optimal earn/hedge pair for a cross-venue carry arb."""
    earn_venue: str
    earn_market: str
    earn_direction: str
    earn_net_carry: float
    hedge_venue: str
    hedge_market: str
    hedge_direction: str
    hedge_net_carry: float
    edge_apr: float                   # earn_net - hedge_net
    # PnL breakdown
    funding_diff_apr: float           # earn.funding_apr - hedge.funding_apr
    basis_diff_apr: float             # (earn.basis or 0) - (hedge.basis or 0)
    total_fees_apr: float             # earn.fee_cost + hedge.fee_cost
    total_slippage_apr: float         # earn.slip_cost + hedge.slip_cost
    pair_expected_apr: float          # funding_diff + basis_diff - fees - slippage
    pair_capacity_usd: float          # min of binding legs
    executable: bool                  # both legs fill within 10bps at size


class CrossVenueResult(BaseModel):
    """Full cross-venue comparison for one symbol."""
    symbol: str
    size_usd: float
    hold_days: int
    venues: List[VenueCarryLeg]
    best_pair: Optional[CrossVenueArbPair]
    computed_at: datetime


class ArbLeaderboardRow(BaseModel):
    """One row in the cross-venue arb leaderboard."""
    rank: int
    symbol: str
    num_venues: int
    earn_venue: str
    earn_direction: str
    earn_net_carry: float
    hedge_venue: str
    hedge_direction: str
    hedge_net_carry: float
    edge_apr: float
    # PnL breakdown
    funding_diff_apr: float = 0.0
    basis_diff_apr: float = 0.0
    total_fees_apr: float = 0.0
    total_slippage_apr: float = 0.0
    pair_expected_apr: float = 0.0
    # Capacity + quality
    pair_capacity_usd: float
    executable: bool
    quality_min: Optional[float]
    trap_tags_union: List[str] = Field(default_factory=list)
    max_data_age_seconds: Optional[float] = None
    earn_data_ts: Optional[datetime] = None
    hedge_data_ts: Optional[datetime] = None
