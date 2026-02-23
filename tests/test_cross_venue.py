"""
tests/test_cross_venue.py
=========================
Unit tests for the cross-venue carry arbitrage logic.
"""

from datetime import datetime, timedelta

import pytest

from core.cross_venue import (
    compute_arb_leaderboard,
    compute_cross_venue_comparison,
    compute_venue_carry_leg,
)
from core.models import (
    CrossVenueArbPair,
    CrossVenueResult,
    DerivedMetrics,
    MarketSnapshot,
    OrderbookLevel,
    Venue,
)


NOW = datetime.utcnow()


def _make_orderbook(mid: float, n: int = 10, spread_bps: float = 2.0):
    """Build symmetric bids / asks around a mid price."""
    half_spread = mid * spread_bps / 20_000  # half spread in price terms
    bids, asks = [], []
    for i in range(n):
        bp = mid - half_spread - i * mid * 0.0001  # each level 1bps deeper
        ap = mid + half_spread + i * mid * 0.0001
        bids.append(OrderbookLevel(price=bp, size=1.0))
        asks.append(OrderbookLevel(price=ap, size=1.0))
    return bids, asks


def _make_snap(
    venue: Venue,
    symbol: str,
    market: str,
    funding_rate: float,
    interval_s: int,
    mark: float,
    index: float,
    oi_usd: float = 1_000_000,
    ob_mid: float = 50000,
    ob_levels: int = 10,
) -> MarketSnapshot:
    bids, asks = _make_orderbook(ob_mid, ob_levels)
    return MarketSnapshot(
        venue=venue,
        symbol=symbol,
        market=market,
        ts=NOW,
        mark_price=mark,
        index_price=index,
        funding_rate=funding_rate,
        funding_interval_seconds=interval_s,
        open_interest_usd=oi_usd,
        orderbook_bids=bids,
        orderbook_asks=asks,
        next_funding_time=NOW + timedelta(hours=1),
        orderbook_stale=False,
    )


def _make_metrics(venue: Venue, symbol: str, quality: float = 75.0) -> DerivedMetrics:
    return DerivedMetrics(
        venue=venue,
        symbol=symbol,
        market=f"{symbol}USDT",
        ts=NOW,
        funding_apr=0.0,
        basis=0.0,
        basis_apr=0.0,
        gross_carry_apr=0.0,
        quality_score=quality,
        crowding_score=30.0,
        trap_tags=[],
        carry_direction="SHORT",
    )



class TestVenueCarryLeg:
    """Tests for single-venue carry computation."""

    def test_binance_positive_funding(self):
        """Positive funding → SHORT earns. Basis from mark/index premium."""
        snap = _make_snap(
            Venue.BINANCE, "BTC", "BTCUSDT",
            funding_rate=0.0001, interval_s=28800,
            mark=84250, index=84000, ob_mid=84000,
        )
        leg = compute_venue_carry_leg(snap, None, size_usd=25000, taker_fee_bps=4, now=NOW)

        assert leg.venue == "binance"
        assert leg.carry_direction == "SHORT"
        assert leg.funding_apr > 0
        assert leg.basis_apr is not None
        assert leg.basis_apr > 0  # mark > index → positive basis
        assert leg.gross_carry_apr == pytest.approx(leg.funding_apr + leg.basis_apr)
        assert leg.net_carry_apr < leg.gross_carry_apr  # costs reduce carry
        assert leg.cost_apr > 0

    def test_dydx_basis_is_none(self):
        """dYdX has no separate mark price → basis_apr must be None."""
        snap = _make_snap(
            Venue.DYDX, "BTC", "BTC-USD",
            funding_rate=0.00005, interval_s=3600,
            mark=84000, index=84000, ob_mid=84000,
        )
        leg = compute_venue_carry_leg(snap, None, size_usd=25000, taker_fee_bps=5, now=NOW)

        assert leg.basis_apr is None
        assert leg.gross_carry_apr == leg.funding_apr  # funding only, no basis
        assert leg.venue == "dydx"

    def test_negative_funding_long_direction(self):
        """Negative funding → LONG earns."""
        snap = _make_snap(
            Venue.HYPERLIQUID, "DOGE", "DOGE",
            funding_rate=-0.0002, interval_s=3600,
            mark=0.15, index=0.15, ob_mid=0.15,
        )
        leg = compute_venue_carry_leg(snap, None, size_usd=5000, taker_fee_bps=2.5, now=NOW)

        assert leg.carry_direction == "LONG"
        # calc_funding_apr(-0.0002, 3600) = -0.0002 * (31536000/3600) = -1.752
        # Funding APR is negative when shorts pay longs (long earns).
        # The sign reflects direction: negative = longs earn, positive = shorts earn.
        assert leg.funding_apr < 0

    def test_capacity_both_sides(self):
        """Both buy and sell capacity are computed separately."""
        snap = _make_snap(
            Venue.BINANCE, "ETH", "ETHUSDT",
            funding_rate=0.0001, interval_s=28800,
            mark=3000, index=2990, ob_mid=3000,
        )
        leg = compute_venue_carry_leg(snap, None, size_usd=10000, taker_fee_bps=4, now=NOW)

        assert leg.capacity_sell_10bps > 0
        assert leg.capacity_buy_10bps > 0

    def test_quality_from_metrics(self):
        """Quality score pulled from pre-computed metrics."""
        snap = _make_snap(
            Venue.BINANCE, "SOL", "SOLUSDT",
            funding_rate=0.0003, interval_s=28800,
            mark=180, index=179, ob_mid=180,
        )
        metrics = _make_metrics(Venue.BINANCE, "SOL", quality=82.0)
        leg = compute_venue_carry_leg(snap, metrics, size_usd=25000, taker_fee_bps=4, now=NOW)

        assert leg.quality_score == 82.0

    def test_data_age_seconds(self):
        """Data age computed from snap.ts to now."""
        old_ts = NOW - timedelta(seconds=15)
        snap = _make_snap(
            Venue.BINANCE, "BTC", "BTCUSDT",
            funding_rate=0.0001, interval_s=28800,
            mark=84000, index=84000, ob_mid=84000,
        )
        snap.ts = old_ts
        leg = compute_venue_carry_leg(snap, None, size_usd=25000, taker_fee_bps=4, now=NOW)

        assert leg.data_age_seconds == pytest.approx(15.0, abs=1.0)



class TestCrossVenueComparison:
    """Tests for cross-venue pair finding."""

    def _build_legs(self, size: float = 25000):
        """Build 3 venue legs for BTC."""
        snaps = [
            _make_snap(Venue.BINANCE, "BTC", "BTCUSDT",
                       0.0001, 28800, 84250, 84000, ob_mid=84000, ob_levels=20),
            _make_snap(Venue.HYPERLIQUID, "BTC", "BTC",
                       0.00005, 3600, 84200, 84100, ob_mid=84100, ob_levels=20),
            _make_snap(Venue.DYDX, "BTC", "BTC-USD",
                       0.00003, 3600, 84000, 84000, ob_mid=84000, ob_levels=20),
        ]
        fees = {"binance": 4, "hyperliquid": 2.5, "dydx": 5}
        legs = []
        for snap in snaps:
            fee = fees[snap.venue.value]
            legs.append(compute_venue_carry_leg(snap, None, size, fee, now=NOW))
        return legs

    def test_best_pair_found(self):
        """With 3 venues, a best pair should be identified."""
        legs = self._build_legs()
        result = compute_cross_venue_comparison("BTC", legs, 25000)

        assert result.best_pair is not None
        assert result.best_pair.edge_apr > 0
        assert result.best_pair.earn_venue != result.best_pair.hedge_venue

    def test_edge_is_earn_minus_hedge(self):
        """Edge = earn.net_carry - hedge.net_carry."""
        legs = self._build_legs()
        result = compute_cross_venue_comparison("BTC", legs, 25000)
        pair = result.best_pair

        expected_edge = pair.earn_net_carry - pair.hedge_net_carry
        assert pair.edge_apr == pytest.approx(expected_edge, abs=1e-10)

    def test_single_venue_no_pair(self):
        """Single venue → no pair possible."""
        snap = _make_snap(Venue.BINANCE, "BTC", "BTCUSDT",
                          0.0001, 28800, 84000, 84000, ob_mid=84000)
        leg = compute_venue_carry_leg(snap, None, 25000, 4, now=NOW)
        result = compute_cross_venue_comparison("BTC", [leg], 25000)

        assert result.best_pair is None

    def test_pair_capacity_is_min_of_both_legs(self):
        """Pair capacity = min(earn_side_cap, hedge_side_cap)."""
        legs = self._build_legs(size=1000)  # small size so all legs executable
        result = compute_cross_venue_comparison("BTC", legs, 1000)
        pair = result.best_pair

        # Find the earn and hedge legs
        earn_leg = next(l for l in legs if l.venue == pair.earn_venue)
        hedge_leg = next(l for l in legs if l.venue == pair.hedge_venue)

        if earn_leg.carry_direction == "SHORT":
            earn_cap = earn_leg.capacity_sell_10bps
        else:
            earn_cap = earn_leg.capacity_buy_10bps

        if hedge_leg.carry_direction == "SHORT":
            hedge_cap = hedge_leg.capacity_buy_10bps
        else:
            hedge_cap = hedge_leg.capacity_sell_10bps

        assert pair.pair_capacity_usd == pytest.approx(min(earn_cap, hedge_cap))



class TestArbLeaderboard:
    """Tests for leaderboard ranking."""

    def _make_result(self, symbol: str, earn_net: float, hedge_net: float):
        """Build a CrossVenueResult with a synthetic pair."""
        pair = CrossVenueArbPair(
            earn_venue="binance", earn_market=f"{symbol}USDT",
            earn_direction="SHORT", earn_net_carry=earn_net,
            hedge_venue="dydx", hedge_market=f"{symbol}-USD",
            hedge_direction="SHORT", hedge_net_carry=hedge_net,
            edge_apr=earn_net - hedge_net,
            funding_diff_apr=earn_net - hedge_net,
            basis_diff_apr=0.0,
            total_fees_apr=0.0,
            total_slippage_apr=0.0,
            pair_expected_apr=earn_net - hedge_net,
            pair_capacity_usd=50000, executable=True,
        )
        return CrossVenueResult(
            symbol=symbol, size_usd=25000, hold_days=30,
            venues=[], best_pair=pair, computed_at=NOW,
        )

    def test_sorted_by_edge_descending(self):
        """Leaderboard rows are ranked by edge_apr desc."""
        results = [
            self._make_result("BTC", 0.20, 0.05),   # edge 0.15
            self._make_result("ETH", 0.30, 0.02),   # edge 0.28
            self._make_result("SOL", 0.10, 0.08),   # edge 0.02
        ]
        rows = compute_arb_leaderboard(results)

        assert len(rows) == 3
        assert rows[0].symbol == "ETH"
        assert rows[1].symbol == "BTC"
        assert rows[2].symbol == "SOL"
        assert rows[0].rank == 1
        assert rows[2].rank == 3

    def test_min_edge_filter(self):
        """min_edge filters out low-edge symbols."""
        results = [
            self._make_result("BTC", 0.20, 0.05),   # edge 0.15
            self._make_result("DOGE", 0.03, 0.02),  # edge 0.01
        ]
        rows = compute_arb_leaderboard(results, min_edge=0.05)

        assert len(rows) == 1
        assert rows[0].symbol == "BTC"



class TestCostSplit:
    """Verify fee/slip/borrow components sum to cost_apr."""

    def test_cost_components_sum_to_total(self):
        snap = _make_snap(
            Venue.BINANCE, "BTC", "BTCUSDT",
            funding_rate=0.0003, interval_s=28800,
            mark=42000, index=42010, ob_mid=42000,
        )
        leg = compute_venue_carry_leg(snap, None, size_usd=25000, taker_fee_bps=4, now=NOW)
        assert leg.fee_cost_apr + leg.slip_cost_apr + leg.borrow_apr == pytest.approx(leg.cost_apr)

    def test_borrow_apr_passed_through(self):
        snap = _make_snap(
            Venue.BINANCE, "BTC", "BTCUSDT",
            funding_rate=0.0003, interval_s=28800,
            mark=42000, index=42010, ob_mid=42000,
        )
        leg = compute_venue_carry_leg(
            snap, None, size_usd=25000, taker_fee_bps=4, borrow_apr=0.05, now=NOW,
        )
        assert leg.borrow_apr == 0.05
        assert leg.fee_cost_apr + leg.slip_cost_apr + 0.05 == pytest.approx(leg.cost_apr)


class TestPairBreakdown:
    """Verify PnL breakdown fields on CrossVenueArbPair."""

    def test_pair_expected_is_gross_diff_minus_both_costs(self):
        """pair_expected_apr = funding_diff + basis_diff - ALL costs (both legs).
        This differs from edge_apr which is earn.net - hedge.net (a spread)."""
        legs = []
        for v, sym, mkt, fee, interval in [
            (Venue.BINANCE, "BTC", "BTCUSDT", 4, 28800),
            (Venue.HYPERLIQUID, "BTC", "BTC", 2.5, 3600),
        ]:
            snap = _make_snap(v, sym, mkt, funding_rate=0.0003, interval_s=interval,
                              mark=42000, index=42010, ob_mid=42000)
            legs.append(compute_venue_carry_leg(snap, None, 25000, fee, now=NOW))

        result = compute_cross_venue_comparison("BTC", legs, 25000, 30)
        pair = result.best_pair
        assert pair is not None

        # pair_expected = funding_diff + basis_diff - total_fees - total_slip
        expected = (pair.funding_diff_apr + pair.basis_diff_apr
                    - pair.total_fees_apr - pair.total_slippage_apr)
        assert pair.pair_expected_apr == pytest.approx(expected, abs=1e-10)

        # edge_apr = earn.net - hedge.net (different formula: spread, not combined PnL)
        earn = next(l for l in legs if l.venue == pair.earn_venue)
        hedge = next(l for l in legs if l.venue == pair.hedge_venue)
        assert pair.edge_apr == pytest.approx(earn.net_carry_apr - hedge.net_carry_apr)

    def test_breakdown_fields_populated(self):
        """Breakdown fields should be non-None floats."""
        legs = []
        for v, sym, mkt, fee, interval in [
            (Venue.BINANCE, "BTC", "BTCUSDT", 4, 28800),
            (Venue.HYPERLIQUID, "BTC", "BTC", 2.5, 3600),
        ]:
            snap = _make_snap(v, sym, mkt, funding_rate=0.0001, interval_s=interval,
                              mark=42000, index=42010, ob_mid=42000)
            legs.append(compute_venue_carry_leg(snap, None, 25000, fee, now=NOW))

        result = compute_cross_venue_comparison("BTC", legs, 25000, 30)
        pair = result.best_pair
        assert pair is not None
        assert isinstance(pair.funding_diff_apr, float)
        assert isinstance(pair.basis_diff_apr, float)
        assert isinstance(pair.total_fees_apr, float)
        assert pair.total_fees_apr > 0  # fees can't be zero with taker fees
        assert isinstance(pair.total_slippage_apr, float)
        assert isinstance(pair.pair_expected_apr, float)


class TestFundingOnlyMode:
    """Verify funding_only flag nullifies basis for all venues."""

    def test_binance_basis_nullified(self):
        """Binance (which normally has basis) should have None when funding_only=True."""
        snap = _make_snap(
            Venue.BINANCE, "BTC", "BTCUSDT",
            funding_rate=0.0003, interval_s=28800,
            mark=42000, index=42010, ob_mid=42000,
        )
        leg = compute_venue_carry_leg(
            snap, None, size_usd=25000, taker_fee_bps=4, now=NOW, funding_only=True,
        )
        assert leg.basis_apr is None
        assert leg.gross_carry_apr == leg.funding_apr

    def test_binance_has_basis_by_default(self):
        """Sanity: Binance normally DOES have basis_apr."""
        snap = _make_snap(
            Venue.BINANCE, "BTC", "BTCUSDT",
            funding_rate=0.0003, interval_s=28800,
            mark=42000, index=42010, ob_mid=42000,
        )
        leg = compute_venue_carry_leg(snap, None, size_usd=25000, taker_fee_bps=4, now=NOW)
        assert leg.basis_apr is not None

    def test_cross_venue_funding_only_levels_playing_field(self):
        """In funding_only mode, all venues should have basis_apr=None."""
        legs = []
        for v, sym, mkt, fee, interval in [
            (Venue.BINANCE, "BTC", "BTCUSDT", 4, 28800),
            (Venue.HYPERLIQUID, "BTC", "BTC", 2.5, 3600),
            (Venue.DYDX, "BTC", "BTC-USD", 5, 3600),
        ]:
            snap = _make_snap(v, sym, mkt, funding_rate=0.0002, interval_s=interval,
                              mark=42000, index=42010, ob_mid=42000)
            legs.append(compute_venue_carry_leg(
                snap, None, 25000, fee, now=NOW, funding_only=True,
            ))
        for leg in legs:
            assert leg.basis_apr is None

    def test_no_pair_excluded(self):
        """Symbols with no best_pair are excluded."""
        no_pair = CrossVenueResult(
            symbol="BNB", size_usd=25000, hold_days=30,
            venues=[], best_pair=None, computed_at=NOW,
        )
        rows = compute_arb_leaderboard([no_pair])
        assert len(rows) == 0


class TestDataTsPropagation:
    """Verify data_ts flows through to leaderboard rows for live age computation."""

    def test_leaderboard_row_has_data_ts(self):
        """earn_data_ts and hedge_data_ts must be set from venue leg timestamps."""
        old_ts = NOW - timedelta(seconds=45)
        snaps = [
            _make_snap(Venue.BINANCE, "BTC", "BTCUSDT",
                       0.0001, 28800, 84250, 84000, ob_mid=84000),
            _make_snap(Venue.DYDX, "BTC", "BTC-USD",
                       0.00003, 3600, 84000, 84000, ob_mid=84000),
        ]
        # Set one snapshot to an older timestamp
        snaps[0].ts = old_ts
        legs = [
            compute_venue_carry_leg(snaps[0], None, 25000, 4, now=NOW),
            compute_venue_carry_leg(snaps[1], None, 25000, 5, now=NOW),
        ]
        result = compute_cross_venue_comparison("BTC", legs, 25000)
        rows = compute_arb_leaderboard([result])

        assert len(rows) == 1
        row = rows[0]
        assert row.earn_data_ts is not None
        assert row.hedge_data_ts is not None
        # The older timestamp should show up on whichever leg uses it
        all_ts = [row.earn_data_ts, row.hedge_data_ts]
        assert old_ts in all_ts
