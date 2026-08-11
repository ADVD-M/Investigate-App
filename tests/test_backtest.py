"""
tests/test_backtest.py — Unit tests for Phase 3: Backtesting
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest import (
    _cumulative_returns,
    _cagr,
    _annualized_vol,
    _sharpe,
    _max_drawdown,
    _calmar,
    _win_rate,
    _total_return,
    _portfolio_daily_returns,
    run_backtest,
    backtest_stats_table,
    TRADING_DAYS,
)
from data import MarketData


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_market_data(n_assets=3, n_days=500, seed=42) -> MarketData:
    """Build a synthetic MarketData namedtuple — no network calls."""
    rng     = np.random.default_rng(seed)
    tickers = [f"ASSET{i}.NS" for i in range(n_assets)]
    dates   = pd.date_range("2019-01-01", periods=n_days, freq="B")

    daily_ret = pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(n_days, n_assets)),
        columns=tickers, index=dates,
    )
    prices = 100 * (1 + daily_ret).cumprod()

    bm_daily  = pd.Series(rng.normal(0.0003, 0.01, n_days), index=dates, name="^NSEI")
    bm_prices = 100 * (1 + bm_daily).cumprod()

    ann_ret = daily_ret.mean() * TRADING_DAYS
    ann_cov = daily_ret.cov()  * TRADING_DAYS

    return MarketData(
        prices             = prices,
        daily_returns      = daily_ret,
        annualized_returns = ann_ret,
        covariance_matrix  = ann_cov,
        benchmark_prices   = bm_prices,
        benchmark_returns  = bm_daily,
        tickers            = tickers,
        period_years       = n_days / TRADING_DAYS,   # float: ~1.98 for 500 days
        fetched_at         = datetime.now(),
    )


EQUAL_WEIGHTS = {"ASSET0.NS": 1/3, "ASSET1.NS": 1/3, "ASSET2.NS": 1/3}


@pytest.fixture(scope="module")
def md():
    return make_market_data()


@pytest.fixture(scope="module")
def bt(md):
    return run_backtest(EQUAL_WEIGHTS, md)


# ── Metric unit tests ─────────────────────────────────────────────────────────

class TestCumulativeReturns:
    def test_starts_at_zero(self, md):
        cum = _cumulative_returns(md.daily_returns.iloc[:, 0])
        # First value should reflect the first day's return, not 0
        # (day 0 return is not 0, but cumulative starts from first day)
        assert isinstance(cum, pd.Series)

    def test_always_growing_for_positive_returns(self):
        always_pos = pd.Series([0.01] * 100)
        cum = _cumulative_returns(always_pos)
        assert cum.is_monotonic_increasing

    def test_final_value_correct(self):
        daily = pd.Series([0.1, -0.1, 0.2])
        cum   = _cumulative_returns(daily)
        expected_final = (1.1 * 0.9 * 1.2) - 1
        assert abs(cum.iloc[-1] - expected_final) < 1e-9


class TestCAGR:
    def test_known_return(self):
        """10% daily return for 252 days should give very high CAGR."""
        daily = pd.Series([0.001] * TRADING_DAYS)  # ~1 basis point daily
        cagr  = _cagr(daily)
        # (1.001)^252 - 1 ≈ 28.3%
        assert abs(cagr - ((1.001 ** TRADING_DAYS) - 1)) < 0.001

    def test_flat_returns_give_zero_cagr(self):
        daily = pd.Series([0.0] * TRADING_DAYS)
        assert abs(_cagr(daily)) < 1e-9

    def test_negative_returns_give_negative_cagr(self):
        daily = pd.Series([-0.001] * TRADING_DAYS)
        assert _cagr(daily) < 0


class TestAnnualizedVol:
    def test_zero_returns_give_zero_vol(self):
        assert abs(_annualized_vol(pd.Series([0.0] * 100))) < 1e-9

    def test_scales_with_sqrt_time(self):
        """Annualized vol should be daily_std × √252."""
        daily = pd.Series(np.random.default_rng(0).normal(0, 0.01, 252))
        assert abs(_annualized_vol(daily) - daily.std() * np.sqrt(TRADING_DAYS)) < 1e-9

    def test_higher_noise_gives_higher_vol(self):
        rng = np.random.default_rng(42)
        low_vol  = pd.Series(rng.normal(0, 0.005, 252))
        high_vol = pd.Series(rng.normal(0, 0.02,  252))
        assert _annualized_vol(high_vol) > _annualized_vol(low_vol)


class TestSharpe:
    def test_higher_return_gives_higher_sharpe(self):
        s1 = _sharpe(0.15, 0.10, 0.065)
        s2 = _sharpe(0.20, 0.10, 0.065)
        assert s2 > s1

    def test_higher_vol_gives_lower_sharpe(self):
        s1 = _sharpe(0.15, 0.10, 0.065)
        s2 = _sharpe(0.15, 0.20, 0.065)
        assert s2 < s1

    def test_zero_vol_returns_zero(self):
        assert _sharpe(0.15, 0.0, 0.065) == 0.0

    def test_known_value(self):
        # (0.20 - 0.065) / 0.10 = 1.35
        assert abs(_sharpe(0.20, 0.10, 0.065) - 1.35) < 1e-9


class TestMaxDrawdown:
    def test_monotonically_increasing_has_zero_drawdown(self):
        daily = pd.Series([0.01] * 100)
        assert abs(_max_drawdown(daily)) < 1e-6

    def test_negative_returns_give_negative_drawdown(self):
        daily = pd.Series([-0.01] * 50)
        assert _max_drawdown(daily) < 0

    def test_known_drawdown(self):
        """Rise to peak then fall 20%: max drawdown should be ~ -20%."""
        # Simulate: go up 10 days, then drop 20%
        up   = [0.02] * 10           # rises to (1.02)^10 ≈ 1.219
        down = [-0.02] * 10          # falls from peak
        daily = pd.Series(up + down)
        dd = _max_drawdown(daily)
        assert dd < 0
        assert dd > -0.5             # shouldn't be worse than -50%

    def test_returns_float(self):
        daily = pd.Series([-0.01, 0.02, -0.03])
        assert isinstance(_max_drawdown(daily), float)


class TestCalmar:
    def test_positive_return_and_drawdown_gives_positive_calmar(self):
        assert _calmar(0.15, -0.30) > 0

    def test_zero_drawdown_returns_zero(self):
        assert _calmar(0.15, 0.0) == 0.0

    def test_known_value(self):
        # 0.15 / 0.30 = 0.5
        assert abs(_calmar(0.15, -0.30) - 0.5) < 1e-9


class TestWinRate:
    def test_all_positive_is_one(self):
        assert _win_rate(pd.Series([0.01, 0.02, 0.005])) == 1.0

    def test_all_negative_is_zero(self):
        assert _win_rate(pd.Series([-0.01, -0.02])) == 0.0

    def test_half_and_half(self):
        assert abs(_win_rate(pd.Series([0.01, -0.01, 0.01, -0.01])) - 0.5) < 1e-9


class TestTotalReturn:
    def test_known_total_return(self):
        # 1% gain for 3 days: (1.01)^3 - 1 ≈ 0.0303
        daily = pd.Series([0.01, 0.01, 0.01])
        expected = (1.01 ** 3) - 1
        assert abs(_total_return(daily) - expected) < 1e-9

    def test_zero_returns(self):
        assert abs(_total_return(pd.Series([0.0] * 100))) < 1e-9


class TestPortfolioDailyReturns:
    def test_equal_weights_average(self, md):
        w   = EQUAL_WEIGHTS
        ret = _portfolio_daily_returns(w, md.daily_returns)
        # With equal weights, portfolio return ≈ mean of individual returns
        expected = md.daily_returns.mean(axis=1)
        pd.testing.assert_series_equal(ret, expected.rename("Portfolio"), check_names=False)

    def test_single_asset_all_weight(self, md):
        w   = {"ASSET0.NS": 1.0, "ASSET1.NS": 0.0, "ASSET2.NS": 0.0}
        ret = _portfolio_daily_returns(w, md.daily_returns)
        pd.testing.assert_series_equal(
            ret, md.daily_returns["ASSET0.NS"].rename("Portfolio"), check_names=False
        )

    def test_weights_renormalised_when_asset_missing(self, md):
        # Asset not in returns DataFrame should be silently skipped
        w   = {"ASSET0.NS": 0.5, "GHOST.NS": 0.5}
        ret = _portfolio_daily_returns(w, md.daily_returns)
        # All weight goes to ASSET0
        pd.testing.assert_series_equal(
            ret, md.daily_returns["ASSET0.NS"].rename("Portfolio"), check_names=False
        )


# ── run_backtest integration tests ────────────────────────────────────────────

class TestRunBacktest:
    def test_returns_backtest_result(self, bt):
        from backtest import BacktestResult
        assert isinstance(bt, BacktestResult)

    def test_period_years_reasonable(self, md, bt):
        assert 0.5 < bt.period_years <= md.period_years + 0.5

    def test_start_before_end(self, bt):
        assert bt.start_date < bt.end_date

    def test_win_rate_in_range(self, bt):
        assert 0.0 <= bt.win_rate <= 1.0

    def test_cumulative_series_non_empty(self, bt):
        assert len(bt.portfolio_cumulative) > 0
        assert len(bt.benchmark_cumulative) > 0

    def test_lookback_trims_data(self, md):
        """A 1-year backtest should have ~252 data points."""
        bt_1yr = run_backtest(EQUAL_WEIGHTS, md, lookback_years=1)
        assert 200 <= len(bt_1yr.portfolio_cumulative) <= 300

    def test_weights_stored_in_result(self, bt):
        assert bt.weights == EQUAL_WEIGHTS


class TestBacktestStatsTable:
    def test_returns_dataframe(self, bt):
        df = backtest_stats_table(bt)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, bt):
        df = backtest_stats_table(bt)
        assert "Metric"             in df.columns
        assert "Portfolio"          in df.columns
        assert "NIFTY 50 Benchmark" in df.columns

    def test_row_count(self, bt):
        df = backtest_stats_table(bt)
        assert len(df) == 7   # Total Return, CAGR, Vol, Sharpe, MaxDD, Calmar, Win Rate
