"""
core/execution.py
=================
Orderbook fill simulation and capacity calculation.

Sign convention (explicit):
  - funding_rate > 0  → SHORT earns → we SELL the perp → consume bids (best bid downward)
  - funding_rate < 0  → LONG earns  → we BUY  the perp → consume asks (best ask upward)

Slippage direction:
  - SELL side: slippage = (best_bid - avg_fill) / best_bid * 10_000   [positive = worse fill]
  - BUY  side: slippage = (avg_fill - best_ask) / best_ask * 10_000   [positive = worse fill]
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from core.models import (
    CapacityResult,
    FillSide,
    OrderbookLevel,
    SlippagePoint,
    Venue,
)



def simulate_fill(
    side: FillSide,
    notional_usd: float,
    bids: List[OrderbookLevel],   # sorted descending by price
    asks: List[OrderbookLevel],   # sorted ascending  by price
    max_slippage_bps: float = 100.0,
) -> SlippagePoint:
    """
    Walk orderbook levels consuming size until notional_usd is filled
    or max_slippage_bps is breached.

    Returns a SlippagePoint describing the simulated execution.
    slippage_bps = 9999 signals the fill is not feasible (empty book or
    threshold hit at first level).
    """
    if notional_usd <= 0:
        return SlippagePoint(
            size_usd=notional_usd, avg_fill_price=0.0,
            executed_notional=0.0, unfilled_notional=notional_usd,
            slippage_bps=0.0, levels_used=0, fill_complete=True,
        )

    levels = bids if side == FillSide.SELL else asks

    if not levels:
        return SlippagePoint(
            size_usd=notional_usd, avg_fill_price=0.0,
            executed_notional=0.0, unfilled_notional=notional_usd,
            slippage_bps=9999.0, levels_used=0, fill_complete=False,
        )

    reference_price = levels[0].price
    remaining = notional_usd
    total_cost = 0.0
    total_qty  = 0.0
    levels_used = 0

    for level in levels:
        if remaining <= 1e-9:
            break

        # Slippage at this level vs reference price
        if side == FillSide.SELL:
            level_slip_bps = (reference_price - level.price) / reference_price * 10_000
        else:
            level_slip_bps = (level.price - reference_price) / reference_price * 10_000

        # Stop consuming if this level already exceeds threshold
        if level_slip_bps > max_slippage_bps:
            break

        level_notional = level.price * level.size
        fill_notional  = min(remaining, level_notional)
        fill_qty       = fill_notional / level.price

        total_cost += fill_qty * level.price
        total_qty  += fill_qty
        remaining  -= fill_notional
        levels_used += 1

    if total_qty < 1e-12:
        return SlippagePoint(
            size_usd=notional_usd, avg_fill_price=reference_price,
            executed_notional=0.0, unfilled_notional=notional_usd,
            slippage_bps=9999.0, levels_used=0, fill_complete=False,
        )

    avg_fill_price   = total_cost / total_qty
    executed_notional = notional_usd - remaining

    if side == FillSide.SELL:
        slippage_bps = (reference_price - avg_fill_price) / reference_price * 10_000
    else:
        slippage_bps = (avg_fill_price - reference_price) / reference_price * 10_000

    slippage_bps = max(0.0, slippage_bps)

    return SlippagePoint(
        size_usd=notional_usd,
        avg_fill_price=avg_fill_price,
        executed_notional=executed_notional,
        unfilled_notional=remaining,
        slippage_bps=slippage_bps,
        levels_used=levels_used,
        fill_complete=(remaining <= notional_usd * 0.001),
    )



def capacity_at_slippage(
    target_bps: float,
    levels: List[OrderbookLevel],
    side: FillSide,
    reference_price: Optional[float] = None,
) -> float:
    """
    Walk levels until slippage exceeds target_bps.
    Return total USD notional fillable within that slippage.

    Uses the top-of-book as reference price if not supplied.
    """
    if not levels:
        return 0.0

    ref = reference_price if reference_price is not None else levels[0].price
    total_notional = 0.0

    for level in levels:
        if side == FillSide.SELL:
            slip = (ref - level.price) / ref * 10_000
        else:
            slip = (level.price - ref) / ref * 10_000

        if slip > target_bps:
            break

        total_notional += level.price * level.size

    return total_notional



SIZE_GRID_USD: List[float] = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
CAPACITY_TIERS_BPS: List[float] = [5.0, 10.0, 25.0]


def build_capacity_result(
    venue: Venue,
    symbol: str,
    market: str,
    bids: List[OrderbookLevel],
    asks: List[OrderbookLevel],
    size_grid: Optional[List[float]] = None,
) -> CapacityResult:
    """
    Build a full CapacityResult for a market:
      - capacity at 5 / 10 / 25 bps for both sides
      - slippage curves (sell + buy) across size_grid
      - spread
    """
    grid = size_grid or SIZE_GRID_USD

    best_bid = bids[0].price if bids else None
    best_ask = asks[0].price if asks else None
    spread_bps: Optional[float] = None
    if best_bid and best_ask and best_bid > 0:
        spread_bps = (best_ask - best_bid) / best_bid * 10_000

    # Sell-side capacity (bids)
    cap_5_sell  = capacity_at_slippage(5.0,  bids, FillSide.SELL)
    cap_10_sell = capacity_at_slippage(10.0, bids, FillSide.SELL)
    cap_25_sell = capacity_at_slippage(25.0, bids, FillSide.SELL)

    # Buy-side capacity (asks)
    cap_5_buy  = capacity_at_slippage(5.0,  asks, FillSide.BUY)
    cap_10_buy = capacity_at_slippage(10.0, asks, FillSide.BUY)
    cap_25_buy = capacity_at_slippage(25.0, asks, FillSide.BUY)

    # Slippage curves
    sell_curve = [simulate_fill(FillSide.SELL, sz, bids, asks, max_slippage_bps=200)
                  for sz in grid]
    buy_curve  = [simulate_fill(FillSide.BUY,  sz, bids, asks, max_slippage_bps=200)
                  for sz in grid]

    return CapacityResult(
        venue=venue,
        symbol=symbol,
        market=market,
        ts=datetime.utcnow(),
        capacity_5bps=cap_5_sell,
        capacity_10bps=cap_10_sell,
        capacity_25bps=cap_25_sell,
        capacity_5bps_buy=cap_5_buy,
        capacity_10bps_buy=cap_10_buy,
        capacity_25bps_buy=cap_25_buy,
        slippage_curve_sell=sell_curve,
        slippage_curve_buy=buy_curve,
        spread_bps=spread_bps,
        bid_levels=len(bids),
        ask_levels=len(asks),
    )



def slippage_for_size(
    side: FillSide,
    notional_usd: float,
    bids: List[OrderbookLevel],
    asks: List[OrderbookLevel],
) -> float:
    """Return slippage_bps for a given side + size. Returns 9999 if not feasible."""
    result = simulate_fill(side, notional_usd, bids, asks, max_slippage_bps=500)
    return result.slippage_bps


def validate_slippage_monotonicity(
    side: FillSide,
    size_grid: List[float],
    bids: List[OrderbookLevel],
    asks: List[OrderbookLevel],
) -> Tuple[bool, List[Tuple[float, float]]]:
    """
    Verify that slippage is monotonic non-decreasing across the size grid.

    Returns ``(is_monotonic, [(size, slippage_bps), ...])``.
    Useful as a sanity check: slippage should never *decrease* as size grows.
    """
    points: List[Tuple[float, float]] = []
    for sz in sorted(size_grid):
        slip = slippage_for_size(side, sz, bids, asks)
        points.append((sz, slip))

    is_mono = all(
        points[i][1] <= points[i + 1][1] + 1e-9
        for i in range(len(points) - 1)
    )
    return is_mono, points
