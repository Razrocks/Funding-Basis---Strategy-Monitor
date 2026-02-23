"""
tests/test_metrics.py
=====================
Unit tests for core/metrics.py pure calculation functions.
"""

import pytest
from core.metrics import (
    calc_basis,
    calc_basis_apr,
    calc_carry_quality_score,
    calc_fee_cost_apr,
    calc_funding_apr,
    calc_gross_carry_apr,
    calc_net_carry_apr,
    calc_slippage_cost_apr,
    calc_zscore,
    carry_direction,
    get_trap_tags,
    realized_funding_apr,
    rolling_std,
)



class TestCalcFundingApr:
    def test_binance_typical(self):
        """Binance: rate=0.0001, interval=28800 → 10.95% APR"""
        result = calc_funding_apr(0.0001, 28800)
        assert abs(result - 0.10950) < 0.0001

    def test_hyperliquid_typical(self):
        """Hyperliquid: rate=0.00042, interval=3600 → ~367.9% APR"""
        result = calc_funding_apr(0.00042, 3600)
        assert abs(result - 3.6792) < 0.001

    def test_zero_rate(self):
        assert calc_funding_apr(0.0, 3600) == 0.0

    def test_negative_rate(self):
        result = calc_funding_apr(-0.0001, 28800)
        assert result < 0

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            calc_funding_apr(0.0001, 0)

    def test_dydx_1h_interval(self):
        """dYdX 1h interval same as Hyperliquid."""
        result = calc_funding_apr(0.0001, 3600)
        assert abs(result - 0.8760) < 0.001



class TestCalcBasis:
    def test_typical_contango(self):
        """mark=84250, index=84000 → basis ≈ 0.002976"""
        result = calc_basis(84250, 84000)
        assert abs(result - 0.002976) < 1e-5

    def test_zero_basis(self):
        """mark == index → basis = 0"""
        assert calc_basis(84000, 84000) == 0.0

    def test_backwardation(self):
        """mark < index → negative basis"""
        result = calc_basis(83800, 84000)
        assert result < 0

    def test_dydx_oracle_equals_mark(self):
        """dYdX: oraclePrice used for both → basis = 0"""
        assert calc_basis(3210.0, 3210.0) == 0.0

    def test_invalid_index(self):
        with pytest.raises(ValueError):
            calc_basis(84000, 0)



class TestCalcBasisApr:
    def test_1day_horizon(self):
        """basis=0.003, horizon=86400 → apr = 0.003 * 365 = 1.095"""
        result = calc_basis_apr(0.003, 86400)
        assert abs(result - 1.095) < 0.001

    def test_zero_basis(self):
        assert calc_basis_apr(0.0, 86400) == 0.0

    def test_7day_horizon_more_conservative(self):
        b_1d = calc_basis_apr(0.003, 86400)
        b_7d = calc_basis_apr(0.003, 604800)
        assert b_7d < b_1d

    def test_invalid_horizon(self):
        with pytest.raises(ValueError):
            calc_basis_apr(0.003, 0)



class TestCalcGrossCarryApr:
    def test_positive(self):
        assert calc_gross_carry_apr(0.1095, 1.095) == pytest.approx(1.2045)

    def test_zero_basis(self):
        assert calc_gross_carry_apr(0.1095, 0.0) == pytest.approx(0.1095)

    def test_negative_funding(self):
        result = calc_gross_carry_apr(-0.05, 0.02)
        assert result == pytest.approx(-0.03)



class TestCarryDirection:
    def test_positive_funding_is_short(self):
        assert carry_direction(0.0001) == "SHORT"

    def test_negative_funding_is_long(self):
        assert carry_direction(-0.0001) == "LONG"

    def test_zero_is_flat(self):
        assert carry_direction(0.0) == "FLAT"



class TestCostCalculations:
    def test_fee_cost_apr_binance(self):
        """4 bps taker, 2x RT, 30d hold → ~0.97% APR"""
        result = calc_fee_cost_apr(4, hold_days=30)
        assert abs(result - 0.00973) < 0.0001

    def test_slippage_cost_apr(self):
        """5 bps slip, 2x RT, 30d hold"""
        result = calc_slippage_cost_apr(5, hold_days=30)
        expected = (5 * 2 / 10_000) * (365 / 30)
        assert abs(result - expected) < 1e-8

    def test_net_carry_apr_positive(self):
        result = calc_net_carry_apr(0.2, 0.05, slippage_bps=3, taker_fee_bps=4)
        assert result > 0

    def test_net_carry_apr_negative_when_costs_exceed(self):
        """Very high fees eat all carry."""
        result = calc_net_carry_apr(0.05, 0.0, slippage_bps=50, taker_fee_bps=100, hold_days=1)
        assert result < 0

    def test_invalid_hold_days(self):
        with pytest.raises(ValueError):
            calc_fee_cost_apr(4, hold_days=0)



class TestRollingStats:
    def test_rolling_std_none_if_single_value(self):
        assert rolling_std([0.1]) is None

    def test_rolling_std_none_if_empty(self):
        assert rolling_std([]) is None

    def test_rolling_std_correct(self):
        result = rolling_std([1.0, 2.0, 3.0, 4.0, 5.0])
        import numpy as np
        expected = float(np.std([1, 2, 3, 4, 5], ddof=1))
        assert abs(result - expected) < 1e-10

    def test_rolling_std_window(self):
        """Should only use last 3 values."""
        result = rolling_std([100, 200, 1.0, 2.0, 3.0], window=3)
        import numpy as np
        expected = float(np.std([1, 2, 3], ddof=1))
        assert abs(result - expected) < 1e-10


class TestZscore:
    def test_returns_none_if_insufficient_history(self):
        assert calc_zscore(1.5, [1.0, 2.0, 3.0]) is None  # < 10 observations

    def test_returns_none_for_exactly_9(self):
        assert calc_zscore(1.5, list(range(9))) is None

    def test_computed_for_10_or_more(self):
        # history has some variance so z-score is meaningful
        history = [1.0, 1.5, 2.0, 1.2, 0.8, 1.8, 1.1, 0.9, 1.6, 1.3]
        result = calc_zscore(5.0, history)   # 5.0 is well above the mean
        assert result is not None
        assert result > 0

    def test_zero_std_returns_zero(self):
        """All same values → std = 0 → z = 0."""
        history = [1.0] * 20
        result = calc_zscore(1.0, history)
        assert result == 0.0

    def test_negative_z(self):
        # history with variance, value well below mean → negative z
        history = [4.5, 5.0, 5.5, 4.8, 5.2, 4.9, 5.1, 5.3, 4.7, 5.0,
                   5.2, 4.8, 5.1, 4.9, 5.3, 5.0, 4.6, 5.4, 5.0, 4.9]
        result = calc_zscore(1.0, history)   # 1.0 well below mean ~5
        assert result is not None
        assert result < 0



class TestRealizedFundingApr:
    def test_none_on_empty(self):
        assert realized_funding_apr([], 3600) is None

    def test_single_rate(self):
        result = realized_funding_apr([0.0001], 28800)
        assert abs(result - 0.10950) < 0.0001

    def test_mean_of_rates(self):
        rates = [0.0001, 0.0002, 0.0003]  # mean = 0.0002
        result = realized_funding_apr(rates, 28800)
        expected = calc_funding_apr(0.0002, 28800)
        assert abs(result - expected) < 1e-8



class TestCarryQualityScore:
    def test_perfect_score_on_stable(self):
        """Low vol, no z-score, no OI change → high score."""
        score = calc_carry_quality_score(
            gross_carry_apr=0.5,
            funding_std_24h=0.001,
            basis_std_24h=0.0001,
            carry_std_24h=0.001,
            funding_zscore=0.5,
            oi_change_1h_pct=1.0,
        )
        assert score >= 85

    def test_low_score_on_high_cv(self):
        """High volatility CV + extreme z-score + OI shock → low score."""
        score = calc_carry_quality_score(
            gross_carry_apr=0.2,
            funding_std_24h=0.5,   # CV = 0.5/0.2 = 2.5 → big funding penalty (35%)
            basis_std_24h=None,
            carry_std_24h=0.5,
            funding_zscore=4.0,    # extreme z → full zscore penalty (20%)
            oi_change_1h_pct=30.0, # OI shock → full oi penalty (20%)
        )
        # Penalties: 35 + 20 + 20 = 75 points → score ≈ 25
        assert score < 40

    def test_high_zscore_penalises(self):
        """Extreme z-score → lower quality."""
        score_normal = calc_carry_quality_score(0.3, None, None, None, 0.5, None)
        score_extreme = calc_carry_quality_score(0.3, None, None, None, 4.0, None)
        assert score_extreme < score_normal

    def test_score_bounded_0_to_100(self):
        for _ in range(10):
            score = calc_carry_quality_score(0.1, 1.0, 1.0, 1.0, 5.0, 50.0)
            assert 0.0 <= score <= 100.0



class TestGetTrapTags:
    def test_no_tags_on_clean_market(self):
        tags = get_trap_tags(funding_zscore=0.5, oi_change_1h_pct=2.0, basis=0.0)
        assert tags == []

    def test_funding_spike_tag(self):
        tags = get_trap_tags(funding_zscore=3.0)
        assert "Funding spike" in tags

    def test_oi_shock_tag(self):
        tags = get_trap_tags(oi_change_1h_pct=25.0)
        assert "OI shock" in tags

    def test_basis_inversion_tag(self):
        tags = get_trap_tags(basis=-0.005)
        assert "Basis inversion" in tags

    def test_carry_unstable_tag(self):
        tags = get_trap_tags(
            carry_std_24h=0.2,
            gross_carry_apr=0.1,  # CV = 2.0 → above threshold
        )
        assert "Carry unstable" in tags

    def test_multiple_tags(self):
        tags = get_trap_tags(
            funding_zscore=3.5,
            oi_change_1h_pct=30.0,
            basis=-0.003,
        )
        assert len(tags) >= 3
