"""
core/metrics.py
===============
Pure calculation functions — no I/O, no side effects.
All functions accept plain Python numbers/lists and return numbers.
These are the canonical formulas used everywhere in the system.

Funding direction convention (explicit):
  funding_rate > 0 → longs pay shorts → SHORT earns → borrow the perp (sell)
  funding_rate < 0 → shorts pay longs → LONG earns → buy the perp (buy)

Cost amortisation:
  Entry/exit fees and slippage are one-time costs paid at open and close.
  We annualise them by amortising over an assumed hold period (default 30 days).
  net_carry_apr = gross_carry_apr - (fee_bps + slippage_bps) / 10000 * (365d / hold_days)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


SECONDS_PER_YEAR: int = 31_536_000   # 365 * 24 * 3600
DEFAULT_BASIS_HORIZON_SECONDS: int = 86_400   # 1 day
DEFAULT_HOLD_DAYS: int = 30



def calc_funding_apr(funding_rate: float, funding_interval_seconds: int) -> float:
    """
    Annualise a per-interval funding rate.

    Example:
      Binance:      rate=0.0001, interval=28800 → 0.0001 * (31536000/28800) = 0.10950 = 10.95% APR
      Hyperliquid:  rate=0.00042, interval=3600 → 0.00042 * 8760 = 3.6792 = 367.9% APR
    """
    if funding_interval_seconds <= 0:
        raise ValueError(f"funding_interval_seconds must be > 0, got {funding_interval_seconds}")
    return funding_rate * (SECONDS_PER_YEAR / funding_interval_seconds)


def calc_basis(mark_price: float, index_price: float) -> float:
    """
    Perp premium over index.
    basis > 0 → contango (mark above index) → longs pay, shorts earn
    basis < 0 → backwardation (mark below index)

    Note: dYdX has no separate mark price; index_price == mark_price → basis always 0.
    """
    if index_price <= 0:
        raise ValueError(f"index_price must be > 0, got {index_price}")
    return (mark_price / index_price) - 1.0


def calc_basis_apr(basis: float, horizon_seconds: int = DEFAULT_BASIS_HORIZON_SECONDS) -> float:
    """
    Annualise basis assuming it decays to zero over horizon_seconds.
    basis_apr = basis * (seconds_per_year / horizon_seconds)

    Default horizon = 1 day (86400s). Can be set to 7d (604800s) for conservative estimate.
    """
    if horizon_seconds <= 0:
        raise ValueError(f"horizon_seconds must be > 0, got {horizon_seconds}")
    return basis * (SECONDS_PER_YEAR / horizon_seconds)


def calc_gross_carry_apr(funding_apr: float, basis_apr: float) -> float:
    """Gross carry = funding component + basis component (both annualised)."""
    return funding_apr + basis_apr


def carry_direction(funding_rate: float) -> str:
    """
    Returns the perp side that earns funding.
    Positive funding → SHORT earns.
    Negative funding → LONG earns.
    Zero → no carry directional edge.
    """
    if funding_rate > 0:
        return "SHORT"
    if funding_rate < 0:
        return "LONG"
    return "FLAT"



def calc_fee_cost_apr(
    taker_fee_bps: float,
    hold_days: int = DEFAULT_HOLD_DAYS,
    round_trip_multiplier: int = 2,
) -> float:
    """
    Annualise round-trip taker fee cost over assumed hold period.

    fee_cost_apr = (taker_fee_bps * round_trip_multiplier / 10000) * (365 / hold_days)

    Example: 4 bps taker, 2x round trip, 30d hold
      = (4 * 2 / 10000) * (365/30) = 0.0008 * 12.167 = 0.00973 = 0.97% APR
    """
    if hold_days <= 0:
        raise ValueError(f"hold_days must be > 0, got {hold_days}")
    fee_fraction = (taker_fee_bps * round_trip_multiplier) / 10_000
    return fee_fraction * (365 / hold_days)


def calc_slippage_cost_apr(
    slippage_bps: float,
    hold_days: int = DEFAULT_HOLD_DAYS,
    round_trip_multiplier: int = 2,
) -> float:
    """
    Annualise round-trip slippage cost over assumed hold period.
    Slippage is paid on entry AND exit (hence round_trip_multiplier=2).
    """
    if hold_days <= 0:
        raise ValueError(f"hold_days must be > 0, got {hold_days}")
    slip_fraction = (slippage_bps * round_trip_multiplier) / 10_000
    return slip_fraction * (365 / hold_days)


def calc_net_carry_apr(
    funding_apr: float,
    basis_apr: float,
    slippage_bps: float,
    taker_fee_bps: float,
    borrow_apr: float = 0.0,
    hold_days: int = DEFAULT_HOLD_DAYS,
) -> float:
    """
    Net carry after all costs.

    net_carry = gross_carry - fee_cost_apr - slippage_cost_apr - borrow_apr

    Args:
        funding_apr:    annualised funding rate
        basis_apr:      annualised basis
        slippage_bps:   one-way slippage in basis points (will be 2x for round trip)
        taker_fee_bps:  one-way taker fee in basis points (will be 2x for round trip)
        borrow_apr:     annual borrow rate on collateral (default 0)
        hold_days:      assumed hold period for amortising entry/exit costs
    """
    gross = calc_gross_carry_apr(funding_apr, basis_apr)
    fee_cost = calc_fee_cost_apr(taker_fee_bps, hold_days)
    slip_cost = calc_slippage_cost_apr(slippage_bps, hold_days)
    return gross - fee_cost - slip_cost - borrow_apr



def rolling_std(values: List[float], window: Optional[int] = None) -> Optional[float]:
    """
    Sample standard deviation of the last `window` values.
    Returns None if fewer than 2 values available.

    Args:
        values: time-ordered list (oldest first)
        window: max number of recent values to use (None = use all)
    """
    if len(values) < 2:
        return None
    arr = np.array(values if window is None else values[-window:], dtype=float)
    if len(arr) < 2:
        return None
    return float(np.std(arr, ddof=1))


def rolling_mean(values: List[float], window: Optional[int] = None) -> Optional[float]:
    """Mean of the last `window` values. Returns None if empty."""
    if not values:
        return None
    arr = np.array(values if window is None else values[-window:], dtype=float)
    return float(np.mean(arr))


def calc_zscore(value: float, history: List[float]) -> Optional[float]:
    """
    Z-score of `value` relative to `history`.
    Returns None if fewer than 10 observations (insufficient for meaningful z-score).
    """
    if len(history) < 10:
        return None
    arr = np.array(history, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float((value - mean) / std)


def realized_funding_apr(
    funding_rates: List[float],
    funding_interval_seconds: int,
) -> Optional[float]:
    """
    Compute realized funding APR from a list of historical per-interval rates.
    Takes the mean of recent rates and annualises.

    Args:
        funding_rates: list of per-interval rates (oldest first)
        funding_interval_seconds: seconds per funding period
    """
    if not funding_rates:
        return None
    mean_rate = float(np.mean(funding_rates))
    return calc_funding_apr(mean_rate, funding_interval_seconds)



def calc_carry_quality_score(
    gross_carry_apr: float,
    funding_std_24h: Optional[float],
    basis_std_24h: Optional[float],
    carry_std_24h: Optional[float],
    funding_zscore: Optional[float],
    oi_change_1h_pct: Optional[float],
    weights: Optional[dict] = None,
) -> float:
    """
    Score from 0 to 100 representing carry quality.
    Higher = more stable, less crowded, more likely to persist.
    Lower = spike-driven, crowded, likely to mean-revert.

    Scoring logic:
      Start at 100.
      Subtract penalties for:
        1. High short-term funding volatility (CV = std/|mean|)
        2. High basis volatility
        3. Extreme z-score (carry likely to revert)
        4. Sudden OI change (crowding inflow / sharp exit)

    Weights (default from plan):
      funding_stability: 0.35
      basis_stability:   0.25
      oi_stability:      0.20
      zscore_penalty:    0.20
    """
    w = weights or {
        "funding_stability": 0.35,
        "basis_stability":   0.25,
        "oi_stability":      0.20,
        "zscore_penalty":    0.20,
    }

    score = 100.0
    abs_carry = abs(gross_carry_apr)

    # Penalty 1: Funding volatility (CV = std / |mean|)
    if funding_std_24h is not None and abs_carry > 1e-6:
        cv = funding_std_24h / abs_carry
        penalty_pct = min(1.0, cv)
        score -= penalty_pct * 100 * w["funding_stability"]

    # Penalty 2: Basis volatility
    if basis_std_24h is not None and abs_carry > 1e-6:
        basis_cv = basis_std_24h * (SECONDS_PER_YEAR / DEFAULT_BASIS_HORIZON_SECONDS) / (abs_carry + 1e-6)
        penalty_pct = min(1.0, basis_cv)
        score -= penalty_pct * 100 * w["basis_stability"]

    # Penalty 3: Z-score extremity (|z| > 1.5 → linear penalty, capped at 4.0)
    if funding_zscore is not None:
        z_abs = abs(funding_zscore)
        if z_abs > 1.5:
            z_penalty_pct = min(1.0, (z_abs - 1.5) / 2.5)
            score -= z_penalty_pct * 100 * w["zscore_penalty"]

    # Penalty 4: OI change (>5% in 1h triggers, capped at 20%)
    if oi_change_1h_pct is not None:
        oi_abs = abs(oi_change_1h_pct)
        if oi_abs > 5.0:
            oi_penalty_pct = min(1.0, (oi_abs - 5.0) / 15.0)
            score -= oi_penalty_pct * 100 * w["oi_stability"]

    return max(0.0, min(100.0, score))



def calc_crowding_score(
    funding_zscore: Optional[float],
    oi_change_24h_pct: Optional[float],
    carry_std_7d: Optional[float],
    gross_carry_apr: float,
) -> float:
    """
    Proxy for how crowded a carry trade is.
    High crowding = high z-score + rising OI + elevated carry vs history.

    Score 0-100: 100 = maximally crowded (most likely to unwind violently).
    """
    score = 0.0

    if funding_zscore is not None:
        z_abs = abs(funding_zscore)
        score += min(50.0, z_abs * 12.5)

    if oi_change_24h_pct is not None and oi_change_24h_pct > 0:
        score += min(30.0, oi_change_24h_pct * 1.5)

    if carry_std_7d is not None and abs(gross_carry_apr) > 1e-6:
        cv_7d = carry_std_7d / abs(gross_carry_apr)
        score += min(20.0, cv_7d * 20)

    return max(0.0, min(100.0, score))



def get_trap_tags(
    funding_zscore: Optional[float] = None,
    funding_zscore_threshold: float = 2.5,
    oi_change_1h_pct: Optional[float] = None,
    oi_shock_threshold_pct: float = 20.0,
    carry_std_24h: Optional[float] = None,
    gross_carry_apr: float = 0.0,
    carry_vol_cv_threshold: float = 0.5,
    basis: float = 0.0,
    basis_inversion_threshold: float = -0.001,
    crowding_score: Optional[float] = None,
    crowding_threshold: float = 70.0,
) -> List[str]:
    """
    Returns a list of trap risk tags for a market.

    Tags:
      "Funding spike"    — funding z-score above threshold
      "OI shock"         — OI changed >threshold% in 1 hour
      "Carry unstable"   — high carry volatility relative to mean
      "Basis inversion"  — mark below index beyond threshold
      "Crowded"          — crowding score above threshold
    """
    tags: List[str] = []

    if funding_zscore is not None and abs(funding_zscore) > funding_zscore_threshold:
        tags.append("Funding spike")

    if oi_change_1h_pct is not None and abs(oi_change_1h_pct) > oi_shock_threshold_pct:
        tags.append("OI shock")

    if (carry_std_24h is not None
            and abs(gross_carry_apr) > 1e-6
            and (carry_std_24h / abs(gross_carry_apr)) > carry_vol_cv_threshold):
        tags.append("Carry unstable")

    if basis < basis_inversion_threshold:
        tags.append("Basis inversion")

    if crowding_score is not None and crowding_score > crowding_threshold:
        tags.append("Crowded")

    return tags



def net_carry_at_sizes(
    funding_apr: float,
    basis_apr: float,
    taker_fee_bps: float,
    slippage_at_size: List[Tuple[float, float]],   # [(size_usd, slippage_bps), ...]
    borrow_apr: float = 0.0,
    hold_days: int = DEFAULT_HOLD_DAYS,
) -> List[Tuple[float, float]]:
    """
    Compute net carry APR at each size in the slippage curve.

    Returns: [(size_usd, net_carry_apr), ...]
    """
    results = []
    for size_usd, slip_bps in slippage_at_size:
        nc = calc_net_carry_apr(
            funding_apr=funding_apr,
            basis_apr=basis_apr,
            slippage_bps=slip_bps,
            taker_fee_bps=taker_fee_bps,
            borrow_apr=borrow_apr,
            hold_days=hold_days,
        )
        results.append((size_usd, nc))
    return results


def calc_breakeven_size(
    size_carry_pairs: List[Tuple[float, float]],
) -> Optional[float]:
    """
    Find the smallest size where net carry <= 0.
    Returns None if net carry is positive at all sizes in the grid.
    """
    for size, nc in size_carry_pairs:
        if nc <= 0:
            return size
    return None


def calc_carry_capacity_integral(
    size_carry_pairs: List[Tuple[float, float]],
) -> float:
    """
    Integral of positive net carry over the size grid.
    Approximated as sum of (net_carry * delta_size) for all size intervals
    where net_carry > 0.

    This is the "capacity-adjusted carry" score: bigger is better.
    Units: APR × USD = dollar-APR (not directly interpretable but useful for ranking).
    """
    total = 0.0
    sizes = [s for s, _ in size_carry_pairs]
    carries = [c for _, c in size_carry_pairs]

    for i in range(len(size_carry_pairs) - 1):
        nc_avg = (carries[i] + carries[i + 1]) / 2
        delta_size = sizes[i + 1] - sizes[i]
        if nc_avg > 0:
            total += nc_avg * delta_size

    return total



def calc_peer_zscore(
    value: float,
    peer_values: List[float],
) -> Optional[float]:
    """
    Z-score of `value` within a peer group (e.g. same-day funding APR across assets).
    Used for cross-market dislocation detection.
    """
    return calc_zscore(value, peer_values)


def detect_outlier_carry(
    symbol: str,
    funding_apr: float,
    peer_funding_aprs: dict,  # {symbol: funding_apr}
    z_threshold: float = 2.0,
) -> Optional[dict]:
    """
    Detect if a symbol's funding APR is an outlier vs its peer group.

    Returns a dict with dislocation details, or None if not an outlier.
    """
    peers = [v for k, v in peer_funding_aprs.items() if k != symbol]
    if len(peers) < 3:
        return None

    z = calc_zscore(funding_apr, peers)
    if z is None:
        return None

    peer_mean = float(np.mean(peers))
    peer_std = float(np.std(peers, ddof=1)) if len(peers) > 1 else 0.0

    if abs(z) >= z_threshold:
        return {
            "symbol": symbol,
            "funding_apr": funding_apr,
            "peer_mean_apr": peer_mean,
            "peer_std_apr": peer_std,
            "z_score": z,
            "direction": "high" if z > 0 else "low",
        }
    return None
