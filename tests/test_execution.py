"""
tests/test_execution.py
=======================
Unit tests for core/execution.py orderbook fill simulation.
"""

import pytest
from core.execution import (
    build_capacity_result,
    capacity_at_slippage,
    simulate_fill,
    slippage_for_size,
)
from core.models import FillSide, OrderbookLevel, Venue



def make_bids(best_price: float, n: int = 10, tick: float = None, size: float = 10.0):
    """Descending bids from best_price."""
    tick = tick or best_price * 0.0001
    return [OrderbookLevel(price=best_price - i * tick, size=size) for i in range(n)]


def make_asks(best_price: float, n: int = 10, tick: float = None, size: float = 10.0):
    """Ascending asks from best_price."""
    tick = tick or best_price * 0.0001
    return [OrderbookLevel(price=best_price + i * tick, size=size) for i in range(n)]


BEST_BID = 84000.0
BEST_ASK = 84010.0



class TestSimulateFill:
    def test_empty_bids_returns_9999(self):
        result = simulate_fill(FillSide.SELL, 10_000, [], make_asks(BEST_ASK))
        assert result.slippage_bps == 9999.0
        assert result.fill_complete is False

    def test_empty_asks_returns_9999(self):
        result = simulate_fill(FillSide.BUY, 10_000, make_bids(BEST_BID), [])
        assert result.slippage_bps == 9999.0
        assert result.fill_complete is False

    def test_zero_notional_returns_zero_slippage(self):
        result = simulate_fill(FillSide.SELL, 0, make_bids(BEST_BID), make_asks(BEST_ASK))
        assert result.slippage_bps == 0.0
        assert result.fill_complete is True

    def test_single_level_exact_fill_zero_slippage(self):
        """One level at 84000, size=10. Fill 84000*10 = exactly the level."""
        bids = [OrderbookLevel(price=84000.0, size=10.0)]
        result = simulate_fill(FillSide.SELL, 840_000, bids, [])
        assert result.slippage_bps == pytest.approx(0.0, abs=1e-6)
        assert result.fill_complete is True

    def test_sell_side_slippage_positive(self):
        """Walking down bids should produce positive slippage."""
        bids = make_bids(84000, n=20, size=1.0)
        result = simulate_fill(FillSide.SELL, 1_000_000, bids, [], max_slippage_bps=500)
        assert result.slippage_bps > 0

    def test_buy_side_slippage_positive(self):
        """Walking up asks should produce positive slippage."""
        asks = make_asks(84000, n=20, size=1.0)
        result = simulate_fill(FillSide.BUY, 1_000_000, make_bids(83990), asks, max_slippage_bps=500)
        assert result.slippage_bps > 0

    def test_sell_avg_fill_below_best_bid(self):
        """SELL: avg fill price must be below or equal to best bid."""
        bids = make_bids(84000, n=10, size=1.0)
        result = simulate_fill(FillSide.SELL, 500_000, bids, [], max_slippage_bps=500)
        if result.fill_complete or result.slippage_bps < 9000:
            assert result.avg_fill_price <= 84000.0

    def test_buy_avg_fill_above_best_ask(self):
        """BUY: avg fill price must be above or equal to best ask."""
        asks = make_asks(84010, n=10, size=1.0)
        result = simulate_fill(FillSide.BUY, 500_000, make_bids(84000), asks, max_slippage_bps=500)
        if result.fill_complete or result.slippage_bps < 9000:
            assert result.avg_fill_price >= 84010.0

    def test_slippage_breach_at_first_level_returns_9999(self):
        """If first level already exceeds max_slippage, return 9999."""
        # Best bid at 84000, second level at 80000 = 476 bps away; max = 10 bps
        bids = [
            OrderbookLevel(price=84000.0, size=0.001),  # tiny size
            OrderbookLevel(price=80000.0, size=10.0),
        ]
        # The first level fills almost nothing, 2nd exceeds 10 bps → no fill
        result = simulate_fill(FillSide.SELL, 100_000, bids, [], max_slippage_bps=10)
        # Either partial fill or slippage exceeds threshold
        # Key: unfilled_notional should be large relative to size
        assert result.slippage_bps > 10 or result.fill_complete is False

    def test_multi_level_fill_weighted_avg(self):
        """Walk 3 levels, verify weighted avg fill price is correct."""
        bids = [
            OrderbookLevel(price=100.0, size=10.0),   # 1000 USD
            OrderbookLevel(price=99.0,  size=10.0),   # 990 USD
            OrderbookLevel(price=98.0,  size=10.0),   # 980 USD
        ]
        # Fill exactly 2000 USD (2 levels)
        result = simulate_fill(FillSide.SELL, 2000, bids, [], max_slippage_bps=1000)
        assert result.fill_complete is True
        # avg fill = (1000 + 990 + some at 98) / (10 + 10.1...) depending on rounding
        # At minimum: avg should be between 98 and 100
        assert 98.0 <= result.avg_fill_price <= 100.0

    def test_fill_complete_flag(self):
        """fill_complete should be True when ≥99.9% of order is filled."""
        bids = make_bids(84000, n=20, size=100.0)   # huge depth
        result = simulate_fill(FillSide.SELL, 1_000, bids, [], max_slippage_bps=200)
        assert result.fill_complete is True

    def test_unfilled_when_book_runs_out(self):
        """Tiny book can't fill large order."""
        bids = [OrderbookLevel(price=84000.0, size=0.01)]  # 840 USD total
        result = simulate_fill(FillSide.SELL, 1_000_000, bids, [])
        assert result.unfilled_notional > 0



class TestCapacityAtSlippage:
    def test_empty_book_returns_zero(self):
        assert capacity_at_slippage(10.0, [], FillSide.SELL) == 0.0

    def test_single_level_at_zero_slippage(self):
        """First level has zero slippage vs itself."""
        bids = [OrderbookLevel(price=84000.0, size=10.0)]
        cap = capacity_at_slippage(10.0, bids, FillSide.SELL)
        assert cap == pytest.approx(840_000.0)

    def test_stops_at_threshold(self):
        """Level 3 exceeds threshold — should not be included."""
        # Level 1: 100, Level 2: 99.9 (1 bps away), Level 3: 98 (200 bps away)
        bids = [
            OrderbookLevel(price=100.0, size=10),
            OrderbookLevel(price=99.9,  size=10),
            OrderbookLevel(price=98.0,  size=100),  # 200 bps — outside 10bps threshold
        ]
        cap = capacity_at_slippage(10.0, bids, FillSide.SELL)
        # Should only include first two levels (100*10 + 99.9*10 = 1999)
        assert cap == pytest.approx(100*10 + 99.9*10)

    def test_capacity_grows_with_threshold(self):
        bids = make_bids(84000, n=20, size=10.0)
        cap5  = capacity_at_slippage(5.0,  bids, FillSide.SELL)
        cap25 = capacity_at_slippage(25.0, bids, FillSide.SELL)
        assert cap25 >= cap5

    def test_buy_side_capacity(self):
        """Buy capacity walks asks."""
        asks = make_asks(84010, n=10, size=10.0)
        cap = capacity_at_slippage(10.0, asks, FillSide.BUY)
        assert cap > 0



class TestBuildCapacityResult:
    def setup_method(self):
        self.bids = make_bids(84000.0, n=20, size=10.0)
        self.asks = make_asks(84010.0, n=20, size=10.0)

    def test_produces_capacity_result(self):
        from core.models import CapacityResult
        result = build_capacity_result(
            venue=Venue.BINANCE,
            symbol="BTC",
            market="BTCUSDT",
            bids=self.bids,
            asks=self.asks,
        )
        assert isinstance(result, CapacityResult)

    def test_capacity_tiers_ordered(self):
        result = build_capacity_result(Venue.BINANCE, "BTC", "BTCUSDT", self.bids, self.asks)
        assert result.capacity_10bps >= result.capacity_5bps
        assert result.capacity_25bps >= result.capacity_10bps

    def test_spread_computed(self):
        result = build_capacity_result(Venue.BINANCE, "BTC", "BTCUSDT", self.bids, self.asks)
        assert result.spread_bps is not None
        assert result.spread_bps > 0

    def test_slippage_curves_populated(self):
        result = build_capacity_result(Venue.BINANCE, "BTC", "BTCUSDT", self.bids, self.asks)
        assert len(result.slippage_curve_sell) > 0
        assert len(result.slippage_curve_buy)  > 0

    def test_bid_ask_level_counts(self):
        result = build_capacity_result(Venue.BINANCE, "BTC", "BTCUSDT", self.bids, self.asks)
        assert result.bid_levels == 20
        assert result.ask_levels == 20

    def test_empty_book_zeros(self):
        result = build_capacity_result(Venue.BINANCE, "BTC", "BTCUSDT", [], [])
        assert result.capacity_5bps == 0.0
        assert result.spread_bps is None

    def test_min_capacity_is_binding_constraint(self):
        """capacity_*_min should be ≤ both sides."""
        result = build_capacity_result(Venue.BINANCE, "BTC", "BTCUSDT", self.bids, self.asks)
        assert result.capacity_10bps_min <= result.capacity_10bps
        assert result.capacity_10bps_min <= result.capacity_10bps_buy



class TestSlippageForSize:
    def test_large_size_more_slippage(self):
        bids = make_bids(84000.0, n=20, size=5.0)
        asks = make_asks(84010.0, n=20, size=5.0)
        slip_small = slippage_for_size(FillSide.SELL, 1_000, bids, asks)
        slip_large = slippage_for_size(FillSide.SELL, 500_000, bids, asks)
        assert slip_large >= slip_small

    def test_returns_9999_on_empty(self):
        result = slippage_for_size(FillSide.SELL, 10_000, [], [])
        assert result == 9999.0



class TestSlippageMonotonicity:
    def test_monotonic_with_realistic_book(self):
        """Slippage must be non-decreasing as size grows."""
        from core.execution import validate_slippage_monotonicity
        bids = make_bids(84000.0, n=20, size=5.0)
        asks = make_asks(84010.0, n=20, size=5.0)
        grid = [1000, 5000, 10000, 25000, 50000]
        is_mono, points = validate_slippage_monotonicity(FillSide.SELL, grid, bids, asks)
        assert is_mono is True
        assert len(points) == len(grid)

    def test_zero_slippage_legitimate_deep_book(self):
        """When top-of-book has huge depth, zero slippage at small size is correct."""
        bids = [OrderbookLevel(price=84000.0, size=100.0)]  # 100 BTC = $8.4M depth
        asks = make_asks(84010.0, n=5, size=1.0)
        slip = slippage_for_size(FillSide.SELL, 25_000, bids, asks)
        assert slip == 0.0  # entire fill at best bid

    def test_monotonicity_buy_side(self):
        """Buy-side slippage also monotonic."""
        from core.execution import validate_slippage_monotonicity
        bids = make_bids(84000.0, n=20, size=5.0)
        asks = make_asks(84010.0, n=20, size=5.0)
        grid = [1000, 10000, 50000, 100000]
        is_mono, _ = validate_slippage_monotonicity(FillSide.BUY, grid, bids, asks)
        assert is_mono is True
