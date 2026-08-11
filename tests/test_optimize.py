"""
tests/test_optimize.py — Unit tests for Phase 2: Optimization Engine
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimize import (
    get_efficient_frontier,
    frontier_to_dataframe,
    weights_to_series,
    PortfolioResult,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_N_POINTS,
)
from data import MarketData
from datetime import datetime


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_synthetic_market_data(n_assets=4, n_days=500, seed=42) -> MarketData:
    """
    Build a synthetic MarketData namedtuple with realistic-ish values.
    Avoids any network calls — safe to run offline.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET{i}.NS" for i in range(n_assets)]
    dates   = pd.date_range("2019-01-01", periods=n_days, freq="B")

    # Generate random daily returns with mild correlations
    daily_rets = pd.DataFrame(
        rng.multivariate_normal(
            mean=np.array([0.0003, 0.0004, 0.0002, 0.0001]),
            cov=np.array([
                [0.0001,  0.00003, 0.00002, 0.00001],
                [0.00003, 0.00015, 0.00002, 0.00001],
                [0.00002, 0.00002, 0.00008, 0.000005],
                [0.00001, 0.00001, 0.000005, 0.00004],
            ]),
            size=n_days,
        ),
        columns=tickers,
        index=dates,
    )

    prices = 100 * (1 + daily_rets).cumprod()
    ann_returns = daily_rets.mean() * 252
    ann_cov     = daily_rets.cov()  * 252

    # Benchmark: simple random walk
    bm_daily = pd.Series(rng.normal(0.0003, 0.01, n_days), index=dates, name="^NSEI")
    bm_prices = 100 * (1 + bm_daily).cumprod()

    return MarketData(
        prices             = prices,
        daily_returns      = daily_rets,
        annualized_returns = ann_returns,
        covariance_matrix  = ann_cov,
        benchmark_prices   = bm_prices,
        benchmark_returns  = bm_daily,
        tickers            = tickers,
        period_years       = 2,
        fetched_at         = datetime.now(),
    )


@pytest.fixture(scope="module")
def md():
    return make_synthetic_market_data()


@pytest.fixture(scope="module")
def frontier(md):
    return get_efficient_frontier(md, n_points=20)


# ── get_efficient_frontier tests ──────────────────────────────────────────────

class TestGetEfficientFrontier:
    def test_frontier_has_points(self, frontier):
        assert len(frontier.frontier_points) > 0

    def test_frontier_point_count_reasonable(self, frontier):
        # Should have computed at least half the requested points
        assert len(frontier.frontier_points) >= 10

    def test_tickers_match_input(self, md, frontier):
        assert set(frontier.tickers) == set(md.tickers)

    def test_risk_free_rate_stored(self, frontier):
        assert frontier.risk_free_rate == DEFAULT_RISK_FREE_RATE


class TestWeightSanity:
    """Weights must sum to ~1.0 and all be in [0, 1] for long-only portfolios."""

    PORTFOLIOS = ["min_volatility", "max_sharpe", "conservative", "balanced", "aggressive"]

    @pytest.mark.parametrize("attr", PORTFOLIOS)
    def test_weights_sum_to_one(self, frontier, attr):
        p = getattr(frontier, attr)
        total = sum(p.weights.values())
        assert abs(total - 1.0) < 0.01, f"{attr}: weights sum to {total:.6f}"

    @pytest.mark.parametrize("attr", PORTFOLIOS)
    def test_weights_non_negative(self, frontier, attr):
        p = getattr(frontier, attr)
        for ticker, w in p.weights.items():
            assert w >= -0.001, f"{attr}: {ticker} has negative weight {w}"

    @pytest.mark.parametrize("attr", PORTFOLIOS)
    def test_weights_at_most_one(self, frontier, attr):
        p = getattr(frontier, attr)
        for ticker, w in p.weights.items():
            assert w <= 1.001, f"{attr}: {ticker} has weight > 1: {w}"


class TestReturnAndVolatility:
    PORTFOLIOS = ["min_volatility", "max_sharpe", "conservative", "balanced", "aggressive"]

    @pytest.mark.parametrize("attr", PORTFOLIOS)
    def test_return_is_positive(self, frontier, attr):
        """Synthetic assets have positive expected returns — all portfolios should too."""
        p = getattr(frontier, attr)
        assert p.expected_return > 0, f"{attr}: negative expected return {p.expected_return}"

    @pytest.mark.parametrize("attr", PORTFOLIOS)
    def test_volatility_is_positive(self, frontier, attr):
        p = getattr(frontier, attr)
        assert p.volatility > 0

    def test_min_vol_has_lowest_volatility(self, frontier):
        """Minimum Volatility portfolio should have the lowest vol on the frontier."""
        min_vol = frontier.min_volatility.volatility
        for pt in frontier.frontier_points:
            assert pt.volatility >= min_vol - 0.001, (
                f"Frontier point has lower vol ({pt.volatility:.4f}) than min-vol ({min_vol:.4f})"
            )

    def test_conservative_lower_vol_than_aggressive(self, frontier):
        """Conservative preset should be less volatile than Aggressive."""
        assert frontier.conservative.volatility < frontier.aggressive.volatility

    def test_conservative_lower_return_than_aggressive(self, frontier):
        """Conservative preset should have lower expected return than Aggressive."""
        assert frontier.conservative.expected_return < frontier.aggressive.expected_return


class TestSharpeRatio:
    def test_max_sharpe_highest_sharpe(self, frontier):
        """Max-Sharpe portfolio should have the highest Sharpe on the frontier."""
        max_sh = frontier.max_sharpe.sharpe_ratio
        for pt in frontier.frontier_points:
            # Allow a small tolerance for numerical precision
            assert pt.sharpe_ratio <= max_sh + 0.05, (
                f"Frontier point Sharpe ({pt.sharpe_ratio:.4f}) > max-Sharpe ({max_sh:.4f})"
            )


# ── frontier_to_dataframe tests ───────────────────────────────────────────────

class TestFrontierToDataframe:
    def test_returns_dataframe(self, frontier):
        df = frontier_to_dataframe(frontier)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, frontier):
        df = frontier_to_dataframe(frontier)
        for col in ["volatility", "expected_return", "sharpe_ratio"]:
            assert col in df.columns

    def test_row_count_matches_frontier_points(self, frontier):
        df = frontier_to_dataframe(frontier)
        assert len(df) == len(frontier.frontier_points)

    def test_volatility_monotonically_increasing(self, frontier):
        """Frontier should be sorted by increasing volatility (left to right on chart)."""
        df = frontier_to_dataframe(frontier)
        vols = df["volatility"].values
        assert np.all(vols[1:] >= vols[:-1] - 1e-6), "Frontier volatility not monotonic"


# ── weights_to_series tests ───────────────────────────────────────────────────

class TestWeightsToSeries:
    def test_returns_series(self, frontier):
        s = weights_to_series(frontier.min_volatility)
        assert isinstance(s, pd.Series)

    def test_filters_negligible_weights(self, frontier):
        """Weights below 0.001 should be excluded from the series."""
        s = weights_to_series(frontier.min_volatility)
        assert all(v > 0.001 for v in s.values)

    def test_with_label_mapping(self, frontier):
        labels = {t: f"Label_{t}" for t in frontier.tickers}
        s = weights_to_series(frontier.min_volatility, ticker_labels=labels)
        for idx in s.index:
            assert idx.startswith("Label_")

    def test_single_asset_edge_case(self):
        """Single-asset portfolio should still work correctly."""
        p = PortfolioResult(
            weights         = {"ASSET0.NS": 1.0},
            expected_return = 0.12,
            volatility      = 0.15,
            sharpe_ratio    = 0.5,
            label           = "Single",
        )
        s = weights_to_series(p)
        assert len(s) == 1
        assert abs(s.iloc[0] - 1.0) < 0.001
