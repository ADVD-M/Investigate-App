"""
tests/test_data.py — Unit tests for Phase 1: Data Layer
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data as data_module
from data import (
    _clean_prices,
    _compute_returns,
    DEFAULT_ASSETS,
    TRADING_DAYS,
    MAX_MISSING_PCT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prices(n_days=252, n_assets=3, seed=42) -> pd.DataFrame:
    """Create a synthetic price DataFrame with no missing values."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    prices = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, size=(n_days, n_assets)), axis=0)
    return pd.DataFrame(prices, index=dates, columns=[f"ASSET{i}.NS" for i in range(n_assets)])


# ── _clean_prices tests ───────────────────────────────────────────────────────

class TestCleanPrices:
    def test_forward_fills_short_gaps(self):
        prices = make_prices()
        # Introduce a 3-day gap in column 0
        prices.iloc[10:13, 0] = np.nan
        cleaned = _clean_prices(prices.copy())
        # All NaNs should be filled
        assert cleaned.isnull().sum().sum() == 0

    def test_drops_asset_with_excessive_missing(self):
        prices = make_prices(n_assets=3)
        # Set 25% of ASSET1 to NaN (above threshold)
        n_missing = int(len(prices) * 0.25)
        prices.iloc[:n_missing, 1] = np.nan
        cleaned = _clean_prices(prices.copy())
        # ASSET1 should be dropped
        assert "ASSET1.NS" not in cleaned.columns
        # ASSET0 and ASSET2 should remain
        assert "ASSET0.NS" in cleaned.columns
        assert "ASSET2.NS" in cleaned.columns

    def test_keeps_asset_within_threshold(self):
        prices = make_prices(n_assets=2)
        # Set 10% of ASSET0 to NaN (below MAX_MISSING_PCT)
        n_missing = int(len(prices) * 0.10)
        prices.iloc[:n_missing, 0] = np.nan
        cleaned = _clean_prices(prices.copy())
        assert "ASSET0.NS" in cleaned.columns

    def test_all_nan_rows_dropped(self):
        prices = make_prices(n_assets=2)
        # Set an entire row to NaN
        prices.iloc[50] = np.nan
        prices.iloc[51] = np.nan
        cleaned = _clean_prices(prices.copy())
        assert cleaned.isnull().any(axis=1).sum() == 0


# ── _compute_returns tests ────────────────────────────────────────────────────

class TestComputeReturns:
    def test_daily_returns_shape(self):
        prices = make_prices(n_days=252, n_assets=3)
        daily_ret, ann_ret, ann_cov = _compute_returns(prices)
        # daily_returns should have one fewer row than prices (first row is NaN → dropped)
        assert len(daily_ret) == len(prices) - 1
        assert daily_ret.shape[1] == 3

    def test_annualized_returns_scaled(self):
        """Annualized return should be roughly TRADING_DAYS × mean daily return."""
        prices = make_prices(n_days=252, n_assets=2)
        daily_ret, ann_ret, _ = _compute_returns(prices)
        expected = daily_ret.mean() * TRADING_DAYS
        pd.testing.assert_series_equal(ann_ret, expected)

    def test_covariance_matrix_symmetric(self):
        prices = make_prices(n_days=252, n_assets=4)
        _, _, cov = _compute_returns(prices)
        # Covariance matrix must be symmetric
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_covariance_matrix_positive_definite(self):
        """A valid covariance matrix must be positive semi-definite (all eigenvalues ≥ 0)."""
        prices = make_prices(n_days=500, n_assets=4)
        _, _, cov = _compute_returns(prices)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-10), f"Non-PSD covariance: {eigenvalues}"

    def test_covariance_annualized(self):
        """Annualized cov = daily cov * TRADING_DAYS."""
        prices = make_prices(n_days=252, n_assets=2)
        daily_ret, _, ann_cov = _compute_returns(prices)
        expected_cov = daily_ret.cov() * TRADING_DAYS
        pd.testing.assert_frame_equal(ann_cov, expected_cov)


# ── DEFAULT_ASSETS sanity check ───────────────────────────────────────────────

class TestDefaultAssets:
    def test_all_tickers_have_ns_suffix(self):
        for ticker in DEFAULT_ASSETS:
            assert ticker.endswith(".NS"), f"{ticker} missing .NS suffix"

    def test_minimum_asset_count(self):
        assert len(DEFAULT_ASSETS) >= 4, "Need at least 4 assets for meaningful diversification"

    def test_all_labels_non_empty(self):
        for ticker, label in DEFAULT_ASSETS.items():
            assert label.strip(), f"{ticker} has an empty label"


# ── Cache staleness logic ─────────────────────────────────────────────────────

class TestCacheStaleness:
    def test_fresh_cache_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_module, "TIMESTAMP_FILE", str(tmp_path / "ts.txt"))
        monkeypatch.setattr(data_module, "CACHE_TTL_HOURS", 24)
        with open(str(tmp_path / "ts.txt"), "w") as f:
            f.write(datetime.now().isoformat())
        assert data_module._cache_is_fresh() is True

    def test_stale_cache_returns_false(self, tmp_path, monkeypatch):
        from datetime import timedelta
        monkeypatch.setattr(data_module, "TIMESTAMP_FILE", str(tmp_path / "ts.txt"))
        monkeypatch.setattr(data_module, "CACHE_TTL_HOURS", 24)
        old_time = datetime.now() - timedelta(hours=25)
        with open(str(tmp_path / "ts.txt"), "w") as f:
            f.write(old_time.isoformat())
        assert data_module._cache_is_fresh() is False

    def test_missing_cache_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_module, "TIMESTAMP_FILE", str(tmp_path / "nonexistent.txt"))
        assert data_module._cache_is_fresh() is False
