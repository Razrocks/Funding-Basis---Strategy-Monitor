"""
core/cross_venue.py
===================
Pure functions for cross-venue carry arbitrage computation.
No I/O — takes snapshots/metrics in, returns comparison objects out.

Composes primitives from core.metrics and core.execution to answer:
  "For a given asset at a given size, which venue pair gives the best
   carry arbitrage edge after all costs?"
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.execution import capacity_at_slippage, slippage_for_size
from core.metrics import (
    calc_basis,
    calc_basis_apr,
    calc_fee_cost_apr,
    calc_funding_apr,
    calc_slippage_cost_apr,
    carry_direction,
)
from core.models import (
    ArbLeaderboardRow,
    CrossVenueArbPair,
    CrossVenueResult,
    DerivedMetrics,
    FillSide,
    MarketSnapshot,
    Venue,
    VenueCarryLeg,
)


DEFAULT_BASIS_HORIZON_SECONDS: int = 86_400



def compute_venue_carry_leg(
    snap: MarketSnapshot,
    metrics: Optional[DerivedMetrics],
    size_usd: float,
    taker_fee_bps: float,
    hold_days: int = 30,
    borrow_apr: float = 0.0,
    now: Optional[datetime] = None,
    funding_only: bool = False,
) -> VenueCarryLeg:
    """
    Compute the full carry breakdown for one venue at a given trade size.

    For dYdX (where mark_price ≈ index_price by design) or when
    ``funding_only=True``:
      - basis_apr is set to None
      - gross_carry = funding_apr only

    Slippage is computed by walking the stored orderbook at ``size_usd``
    on the side that the carry direction requires:
      - SHORT earns → sells → walks bids
      - LONG earns  → buys  → walks asks
    """
    _now = now or datetime.utcnow()

    # --- Funding ---
    f_apr = calc_funding_apr(snap.funding_rate, snap.funding_interval_seconds)
    direction = carry_direction(snap.funding_rate)

    # --- Basis (None for dYdX or funding-only mode) ---
    is_dydx = snap.venue == Venue.DYDX
    if funding_only or is_dydx:
        b_apr: Optional[float] = None
        gross = f_apr  # funding only
    else:
        basis = calc_basis(snap.mark_price, snap.index_price)
        b_apr = calc_basis_apr(basis, DEFAULT_BASIS_HORIZON_SECONDS)
        gross = f_apr + b_apr

    # --- Slippage at chosen size ---
    if direction == "SHORT":
        fill_side = FillSide.SELL
    else:
        fill_side = FillSide.BUY

    slip_bps = slippage_for_size(
        fill_side, size_usd, snap.orderbook_bids, snap.orderbook_asks,
    )

    # --- Costs (split into components) ---
    fee_cost = calc_fee_cost_apr(taker_fee_bps, hold_days)
    slip_cost = calc_slippage_cost_apr(slip_bps, hold_days)
    total_cost = fee_cost + slip_cost + borrow_apr
    net_carry = gross - total_cost

    # --- Both-side capacity ---
    cap_sell = capacity_at_slippage(10.0, snap.orderbook_bids, FillSide.SELL)
    cap_buy = capacity_at_slippage(10.0, snap.orderbook_asks, FillSide.BUY)

    # --- Spread ---
    spread = snap.spread_bps  # property on MarketSnapshot

    # --- Quality / traps from pre-computed metrics ---
    quality = metrics.quality_score if metrics else None
    traps = metrics.trap_tags if metrics else []

    data_age = (_now - snap.ts).total_seconds()

    return VenueCarryLeg(
        venue=snap.venue.value,
        market=snap.market,
        funding_rate=snap.funding_rate,
        funding_interval_seconds=snap.funding_interval_seconds,
        funding_apr=f_apr,
        basis_apr=b_apr,
        gross_carry_apr=gross,
        slippage_bps=slip_bps,
        taker_fee_bps=taker_fee_bps,
        cost_apr=total_cost,
        fee_cost_apr=fee_cost,
        slip_cost_apr=slip_cost,
        borrow_apr=borrow_apr,
        net_carry_apr=net_carry,
        carry_direction=direction,
        capacity_sell_10bps=cap_sell,
        capacity_buy_10bps=cap_buy,
        spread_bps=spread,
        quality_score=quality,
        trap_tags=traps,
        mark_price=snap.mark_price,
        index_price=snap.index_price,
        open_interest_usd=snap.open_interest_usd,
        next_funding_time=snap.next_funding_time,
        data_ts=snap.ts,
        data_age_seconds=max(0.0, data_age),
    )



def compute_cross_venue_comparison(
    symbol: str,
    venue_legs: List[VenueCarryLeg],
    size_usd: float,
    hold_days: int = 30,
) -> CrossVenueResult:
    """
    Given per-venue carry legs for the same asset, find the optimal
    earn / hedge pair and compute the cross-venue edge.

    Best pair logic:
      - earn = venue with the highest net_carry_apr (the "earning" leg)
      - hedge = venue with the lowest net_carry_apr (the "cost" leg)
      - edge  = earn.net_carry - hedge.net_carry
      - For capacity: the earn leg needs capacity on its carry-direction side,
        the hedge leg needs capacity on the opposite side.
      - pair_capacity = min(earn_side_cap, hedge_side_cap)
    """
    best_pair: Optional[CrossVenueArbPair] = None

    if len(venue_legs) >= 2:
        sorted_legs = sorted(venue_legs, key=lambda l: l.net_carry_apr, reverse=True)
        earn = sorted_legs[0]
        hedge = sorted_legs[-1]

        # --- Direction-aware capacity ---
        # Earn leg: if direction is SHORT → needs to sell (bids)
        #           if direction is LONG  → needs to buy (asks)
        if earn.carry_direction == "SHORT":
            earn_cap = earn.capacity_sell_10bps
        else:
            earn_cap = earn.capacity_buy_10bps

        # Hedge leg: takes the opposite side of the earn direction
        # If the carry is SHORT on earn, hedge side buys on its venue
        # (but hedge direction tells us what that venue's carry is —
        #  for the hedge, we need the OPPOSITE of the hedge carry direction)
        if hedge.carry_direction == "SHORT":
            # Hedge venue's carry is SHORT (earns short).
            # As a hedge we'd go LONG on this venue → need buy capacity
            hedge_cap = hedge.capacity_buy_10bps
        else:
            # Hedge venue's carry is LONG → as a hedge we go SHORT → sell capacity
            hedge_cap = hedge.capacity_sell_10bps

        pair_cap = min(earn_cap, hedge_cap)
        executable = (earn_cap >= size_usd) and (hedge_cap >= size_usd)

        # --- PnL breakdown ---
        funding_diff = earn.funding_apr - hedge.funding_apr
        basis_diff = (earn.basis_apr or 0.0) - (hedge.basis_apr or 0.0)
        total_fees = earn.fee_cost_apr + hedge.fee_cost_apr
        total_slip = earn.slip_cost_apr + hedge.slip_cost_apr
        pair_expected = funding_diff + basis_diff - total_fees - total_slip

        best_pair = CrossVenueArbPair(
            earn_venue=earn.venue,
            earn_market=earn.market,
            earn_direction=earn.carry_direction,
            earn_net_carry=earn.net_carry_apr,
            hedge_venue=hedge.venue,
            hedge_market=hedge.market,
            hedge_direction=hedge.carry_direction,
            hedge_net_carry=hedge.net_carry_apr,
            edge_apr=earn.net_carry_apr - hedge.net_carry_apr,
            funding_diff_apr=funding_diff,
            basis_diff_apr=basis_diff,
            total_fees_apr=total_fees,
            total_slippage_apr=total_slip,
            pair_expected_apr=pair_expected,
            pair_capacity_usd=pair_cap,
            executable=executable,
        )

    return CrossVenueResult(
        symbol=symbol,
        size_usd=size_usd,
        hold_days=hold_days,
        venues=venue_legs,
        best_pair=best_pair,
        computed_at=datetime.utcnow(),
    )



def compute_arb_leaderboard(
    cross_venue_results: List[CrossVenueResult],
    min_edge: float = 0.0,
) -> List[ArbLeaderboardRow]:
    """
    Given a list of CrossVenueResult (one per symbol), build a ranked
    leaderboard sorted by edge_apr descending.

    Only includes symbols that have a best_pair with edge >= min_edge.
    """
    rows: List[ArbLeaderboardRow] = []

    for cvr in cross_venue_results:
        pair = cvr.best_pair
        if pair is None:
            continue
        if pair.edge_apr < min_edge:
            continue

        # Quality = min across both legs
        qualities = [l.quality_score for l in cvr.venues if l.quality_score is not None]
        quality_min = min(qualities) if qualities else None

        # Trap tags = union across all legs
        all_tags: List[str] = []
        for leg in cvr.venues:
            all_tags.extend(leg.trap_tags)
        unique_tags = sorted(set(all_tags))

        # Max data age across earn/hedge legs
        earn_leg = next((l for l in cvr.venues if l.venue == pair.earn_venue), None)
        hedge_leg = next((l for l in cvr.venues if l.venue == pair.hedge_venue), None)
        max_age = max(
            earn_leg.data_age_seconds if earn_leg else 0.0,
            hedge_leg.data_age_seconds if hedge_leg else 0.0,
        )

        rows.append(ArbLeaderboardRow(
            rank=0,
            symbol=cvr.symbol,
            num_venues=len(cvr.venues),
            earn_venue=pair.earn_venue,
            earn_direction=pair.earn_direction,
            earn_net_carry=pair.earn_net_carry,
            hedge_venue=pair.hedge_venue,
            hedge_direction=pair.hedge_direction,
            hedge_net_carry=pair.hedge_net_carry,
            edge_apr=pair.edge_apr,
            funding_diff_apr=pair.funding_diff_apr,
            basis_diff_apr=pair.basis_diff_apr,
            total_fees_apr=pair.total_fees_apr,
            total_slippage_apr=pair.total_slippage_apr,
            pair_expected_apr=pair.pair_expected_apr,
            pair_capacity_usd=pair.pair_capacity_usd,
            executable=pair.executable,
            quality_min=quality_min,
            trap_tags_union=unique_tags,
            max_data_age_seconds=max_age,
            earn_data_ts=earn_leg.data_ts if earn_leg else None,
            hedge_data_ts=hedge_leg.data_ts if hedge_leg else None,
        ))

    rows.sort(key=lambda r: r.edge_apr, reverse=True)
    for i, row in enumerate(rows, 1):
        row.rank = i

    return rows
