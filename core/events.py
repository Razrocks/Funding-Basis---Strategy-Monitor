"""
core/events.py
==============
Event detection logic. Pure functions: accept metrics, return Optional[EventRecord].
Called after every polling cycle inside the API background task.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from core.models import EventRecord, EventType, Severity, Venue



def check_funding_spike(
    venue: Venue,
    market: str,
    symbol: str,
    funding_apr: float,
    threshold: float = 2.0,          # 200% APR
    critical_threshold: float = 5.0, # 500% APR
) -> Optional[EventRecord]:
    """Emit event if |funding_apr| exceeds threshold."""
    if abs(funding_apr) <= threshold:
        return None
    severity = Severity.CRITICAL if abs(funding_apr) >= critical_threshold else Severity.WARNING
    return EventRecord(
        ts=datetime.utcnow(), venue=venue, market=market, symbol=symbol,
        event_type=EventType.FUNDING_SPIKE, severity=severity,
        message=f"{symbol} funding APR {funding_apr*100:+.2f}% exceeds threshold {threshold*100:.0f}%",
        details={"funding_apr": funding_apr, "threshold": threshold,
                 "critical_threshold": critical_threshold},
    )


def check_funding_flip(
    venue: Venue,
    market: str,
    symbol: str,
    prev_rate: float,
    curr_rate: float,
) -> Optional[EventRecord]:
    """Emit event when funding rate changes sign."""
    if prev_rate == 0 or curr_rate == 0:
        return None
    if (prev_rate > 0) == (curr_rate > 0):
        return None
    return EventRecord(
        ts=datetime.utcnow(), venue=venue, market=market, symbol=symbol,
        event_type=EventType.FUNDING_FLIP, severity=Severity.INFO,
        message=f"{symbol} funding flipped: {prev_rate*10000:.3f} bps → {curr_rate*10000:.3f} bps",
        details={"prev_rate": prev_rate, "curr_rate": curr_rate},
    )


def check_basis_inversion(
    venue: Venue,
    market: str,
    symbol: str,
    basis: float,
    threshold: float = -0.001,   # -0.1%
) -> Optional[EventRecord]:
    """Emit event when basis goes negative beyond threshold."""
    if basis >= threshold:
        return None
    severity = Severity.CRITICAL if basis < threshold * 3 else Severity.WARNING
    return EventRecord(
        ts=datetime.utcnow(), venue=venue, market=market, symbol=symbol,
        event_type=EventType.BASIS_INVERSION, severity=severity,
        message=f"{symbol} basis inverted: {basis*10000:.2f} bps (threshold: {threshold*10000:.1f} bps)",
        details={"basis": basis, "basis_bps": basis * 10_000, "threshold": threshold},
    )


def check_oi_shock(
    venue: Venue,
    market: str,
    symbol: str,
    oi_now: float,
    oi_prev: float,
    threshold_pct: float = 20.0,
) -> Optional[EventRecord]:
    """Emit event when OI changes more than threshold_pct in one interval."""
    if oi_prev <= 0:
        return None
    change_pct = (oi_now - oi_prev) / oi_prev * 100
    if abs(change_pct) < threshold_pct:
        return None
    severity = Severity.CRITICAL if abs(change_pct) >= threshold_pct * 2 else Severity.WARNING
    direction = "surged" if change_pct > 0 else "dropped"
    return EventRecord(
        ts=datetime.utcnow(), venue=venue, market=market, symbol=symbol,
        event_type=EventType.OI_SHOCK, severity=severity,
        message=f"{symbol} OI {direction} {change_pct:+.1f}% (now ${oi_now/1e6:.1f}M)",
        details={"oi_now": oi_now, "oi_prev": oi_prev, "change_pct": change_pct},
    )


def check_carry_unstable(
    venue: Venue,
    market: str,
    symbol: str,
    carry_std_24h: float,
    gross_carry_apr: float,
    cv_threshold: float = 0.5,
) -> Optional[EventRecord]:
    """Emit event when carry coefficient of variation is too high (unstable)."""
    if abs(gross_carry_apr) < 1e-6:
        return None
    cv = carry_std_24h / abs(gross_carry_apr)
    if cv < cv_threshold:
        return None
    severity = Severity.CRITICAL if cv >= cv_threshold * 2 else Severity.WARNING
    return EventRecord(
        ts=datetime.utcnow(), venue=venue, market=market, symbol=symbol,
        event_type=EventType.CARRY_UNSTABLE, severity=severity,
        message=f"{symbol} carry unstable: CV={cv:.2f} (std={carry_std_24h*100:.2f}% vs mean={gross_carry_apr*100:.2f}%)",
        details={"carry_std_24h": carry_std_24h, "gross_carry_apr": gross_carry_apr, "cv": cv},
    )


def check_regime_shift(
    venue: Venue,
    market: str,
    symbol: str,
    carry_std_now: float,
    carry_std_prev: float,
    multiplier_threshold: float = 2.5,
) -> Optional[EventRecord]:
    """Emit event when carry volatility jumps significantly (regime shift)."""
    if carry_std_prev <= 0:
        return None
    mult = carry_std_now / carry_std_prev
    if mult < multiplier_threshold:
        return None
    return EventRecord(
        ts=datetime.utcnow(), venue=venue, market=market, symbol=symbol,
        event_type=EventType.REGIME_SHIFT, severity=Severity.WARNING,
        message=f"{symbol} carry vol jumped {mult:.1f}x — potential regime shift",
        details={"carry_std_now": carry_std_now, "carry_std_prev": carry_std_prev, "mult": mult},
    )


def check_carry_opportunity(
    venue: Venue,
    market: str,
    symbol: str,
    net_carry_apr: float,
    quality_score: float,
    capacity_10bps: float,
    min_net_carry: float = 0.10,
    min_quality: float = 65.0,
    min_capacity: float = 1_000_000,
) -> Optional[EventRecord]:
    """Emit info event when a market meets minimum carry opportunity criteria."""
    if (net_carry_apr >= min_net_carry
            and quality_score >= min_quality
            and capacity_10bps >= min_capacity):
        return EventRecord(
            ts=datetime.utcnow(), venue=venue, market=market, symbol=symbol,
            event_type=EventType.CARRY_OPPORTUNITY, severity=Severity.INFO,
            message=(f"{symbol}@{venue.value} net carry {net_carry_apr*100:.2f}% APR, "
                     f"quality {quality_score:.0f}, cap ${capacity_10bps/1e6:.1f}M"),
            details={"net_carry_apr": net_carry_apr, "quality_score": quality_score,
                     "capacity_10bps": capacity_10bps},
        )
    return None



def run_all_checks(
    venue: Venue,
    market: str,
    symbol: str,
    funding_apr: float,
    basis: float,
    gross_carry_apr: float,
    net_carry_apr: Optional[float],
    quality_score: Optional[float],
    capacity_10bps: Optional[float],
    oi_now: Optional[float],
    oi_prev: Optional[float],
    carry_std_24h: Optional[float],
    carry_std_prev: Optional[float],
    prev_funding_rate: Optional[float],
    curr_funding_rate: float,
    # Thresholds (from config)
    funding_spike_threshold: float = 2.0,
    basis_inversion_threshold: float = -0.001,
    oi_shock_threshold_pct: float = 20.0,
    carry_vol_cv_threshold: float = 0.5,
) -> List[EventRecord]:
    """Run every detector and return all triggered events."""
    events: List[EventRecord] = []

    def _add(e: Optional[EventRecord]) -> None:
        if e is not None:
            events.append(e)

    _add(check_funding_spike(venue, market, symbol, funding_apr, funding_spike_threshold))
    _add(check_basis_inversion(venue, market, symbol, basis, basis_inversion_threshold))

    if oi_now is not None and oi_prev is not None:
        _add(check_oi_shock(venue, market, symbol, oi_now, oi_prev, oi_shock_threshold_pct))

    if carry_std_24h is not None:
        _add(check_carry_unstable(venue, market, symbol, carry_std_24h,
                                  gross_carry_apr, carry_vol_cv_threshold))

    if carry_std_24h is not None and carry_std_prev is not None:
        _add(check_regime_shift(venue, market, symbol, carry_std_24h, carry_std_prev))

    if prev_funding_rate is not None:
        _add(check_funding_flip(venue, market, symbol, prev_funding_rate, curr_funding_rate))

    if net_carry_apr is not None and quality_score is not None and capacity_10bps is not None:
        _add(check_carry_opportunity(venue, market, symbol, net_carry_apr,
                                     quality_score, capacity_10bps))

    return events
