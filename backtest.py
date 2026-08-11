"""
backtest.py — Phase 3: Backtesting & Performance Analytics
===========================================================
Simulates historical performance of a given portfolio allocation and computes
standard metrics used by professional portfolio managers.

Financial concepts:
- Cumulative return : Total growth of ₹1 invested from the start. e.g. 0.42 = 42% gain.
- CAGR (ann. return): Constant annual growth rate equivalent = (end/start)^(252/days) - 1
- Volatility        : Annualised std-dev of daily returns — the standard risk measure in MPT.
- Sharpe ratio      : (Return − Risk-free rate) / Volatility. Higher = better risk-adj. return.
- Max drawdown      : Worst peak-to-trough loss in the period — measures tail/downside risk.
- Calmar ratio      : CAGR / |Max drawdown|. Higher = better return per unit of drawdown risk.
- Win rate          : Fraction of trading days with positive returns.
"""

import logging
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

TRADING_DAYS     = 252
DEFAULT_RF_RATE  = 0.065   # India 91-day T-bill ~6.5%

# ── Result type ───────────────────────────────────────────────────────────────

BacktestResult = namedtuple(
    "BacktestResult",
    [
        # Time-series
        "portfolio_cumulative",       # pd.Series: cumulative return (0-indexed from start)
        "benchmark_cumulative",       # pd.Series: benchmark cumulative return
        "portfolio_daily_returns",    # pd.Series: daily returns of the portfolio

        # Portfolio metrics
        "annualized_return",          # float: CAGR of the portfolio
        "annualized_volatility",      # float: annualized volatility
        "sharpe_ratio",               # float: risk-adjusted return
        "max_drawdown",               # float: e.g. -0.35 = worst -35% drawdown
        "calmar_ratio",               # float: CAGR / |max_drawdown|
        "total_return",               # float: total return over the full period
        "win_rate",                   # float: fraction of positive days

        # Benchmark metrics (for comparison)
        "benchmark_annualized_return",
        "benchmark_sharpe",
        "benchmark_max_drawdown",
        "benchmark_total_return",

        # Metadata
        "period_years",               # float: actual backtest length in years
        "start_date",                 # datetime: first date in backtest window
        "end_date",                   # datetime: last date in backtest window
        "risk_free_rate",             # float: risk-free rate used
        "weights",                    # dict: {ticker: weight} used for this backtest
    ],
)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns from a series of daily returns.
    Value at each date = total growth since the start.
    Example: 0.25 means ₹1 invested at start is now worth ₹1.25.
    """
    return (1 + daily_returns).cumprod() - 1


def _cagr(daily_returns: pd.Series) -> float:
    """
    Compound Annual Growth Rate.
    Derived from total return over the exact number of trading days observed.
    """
    n_days      = len(daily_returns)
    total_ret   = (1 + daily_returns).prod()           # gross return factor
    cagr        = total_ret ** (TRADING_DAYS / n_days) - 1
    return float(cagr)


def _annualized_vol(daily_returns: pd.Series) -> float:
    """Annualized volatility = daily std-dev × √252."""
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def _sharpe(annualized_return: float, annualized_vol: float, risk_free_rate: float) -> float:
    """
    Sharpe ratio = (return − risk_free) / volatility.
    A Sharpe > 1.0 is generally considered good; > 2.0 is excellent.
    """
    if annualized_vol < 1e-9:
        return 0.0
    return (annualized_return - risk_free_rate) / annualized_vol


def _max_drawdown(daily_returns: pd.Series) -> float:
    """
    Maximum drawdown = worst peak-to-trough loss in the period.
    Computed from the cumulative wealth index (₹1 invested at start).
    Returns a negative float, e.g. -0.35 for a -35% drawdown.
    """
    wealth_index  = (1 + daily_returns).cumprod()
    rolling_peak  = wealth_index.cummax()
    drawdown      = (wealth_index - rolling_peak) / rolling_peak
    return float(drawdown.min())


def _calmar(annualized_return: float, max_dd: float) -> float:
    """Calmar ratio = CAGR / |max drawdown|. Returns 0 if drawdown is 0."""
    if abs(max_dd) < 1e-9:
        return 0.0
    return annualized_return / abs(max_dd)


def _win_rate(daily_returns: pd.Series) -> float:
    """Fraction of days with a positive return."""
    return float((daily_returns > 0).mean())


def _total_return(daily_returns: pd.Series) -> float:
    """Total (cumulative) return over the entire period."""
    return float((1 + daily_returns).prod() - 1)


# ── Portfolio return construction ─────────────────────────────────────────────

def _portfolio_daily_returns(
    weights: dict[str, float],
    daily_returns_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute the daily return of a portfolio as the weighted sum of its assets' returns.

    r_portfolio_t = Σ(weight_i × r_i_t)

    This is a simplification: it assumes daily rebalancing back to the target weights,
    which understates transaction costs but is the standard MPT backtest approach.

    Parameters
    ----------
    weights         : {ticker: weight} dict, weights sum to 1.0
    daily_returns_df: DataFrame of daily returns (from data.py)

    Returns
    -------
    pd.Series of daily portfolio returns.
    """
    # Only use tickers that are both in weights and in the returns DataFrame
    available  = [t for t in weights if t in daily_returns_df.columns and weights[t] > 0.001]
    w_series   = pd.Series({t: weights[t] for t in available})

    # Renormalise weights in case some assets were dropped
    w_series  /= w_series.sum()

    # Dot product: daily return = sum(weight_i * return_i) for each day
    portfolio_ret = daily_returns_df[available].dot(w_series)
    portfolio_ret.name = "Portfolio"
    return portfolio_ret


# ── Public API ────────────────────────────────────────────────────────────────

def run_backtest(
    weights: dict[str, float],
    market_data,                         # MarketData namedtuple from data.py
    lookback_years: int | None = None,   # None = use all available data
    risk_free_rate: float = DEFAULT_RF_RATE,
) -> BacktestResult:
    """
    Simulate historical performance of a given portfolio allocation.

    Parameters
    ----------
    weights        : {ticker: weight} from an optimize.PortfolioResult
    market_data    : MarketData namedtuple from data.get_market_data()
    lookback_years : Number of years to backtest. None = use all available data.
    risk_free_rate : Annual risk-free rate for Sharpe computation.

    Returns
    -------
    BacktestResult namedtuple with all performance metrics and time-series data.
    """
    daily_ret_df = market_data.daily_returns.copy()
    bm_ret_raw   = market_data.benchmark_returns.copy()

    # ── Trim to lookback window ───────────────────────────────────────────────
    if lookback_years is not None:
        cutoff    = daily_ret_df.index[-1] - pd.DateOffset(years=lookback_years)
        daily_ret_df = daily_ret_df[daily_ret_df.index >= cutoff]
        bm_ret_raw   = bm_ret_raw[bm_ret_raw.index >= cutoff]

    # Align benchmark to the asset return dates
    bm_ret = bm_ret_raw.reindex(daily_ret_df.index).ffill().dropna()

    # ── Portfolio daily returns ───────────────────────────────────────────────
    port_ret = _portfolio_daily_returns(weights, daily_ret_df)

    # Align portfolio and benchmark to a common date range
    common_idx  = port_ret.index.intersection(bm_ret.index)
    port_ret    = port_ret.loc[common_idx]
    bm_ret      = bm_ret.loc[common_idx]

    if len(port_ret) == 0:
        raise ValueError("No overlapping dates between portfolio assets and benchmark.")

    # ── Portfolio metrics ─────────────────────────────────────────────────────
    ann_ret  = _cagr(port_ret)
    ann_vol  = _annualized_vol(port_ret)
    sharpe   = _sharpe(ann_ret, ann_vol, risk_free_rate)
    max_dd   = _max_drawdown(port_ret)
    calmar   = _calmar(ann_ret, max_dd)
    tot_ret  = _total_return(port_ret)
    win_r    = _win_rate(port_ret)

    # ── Benchmark metrics ─────────────────────────────────────────────────────
    bm_ann_ret  = _cagr(bm_ret)
    bm_ann_vol  = _annualized_vol(bm_ret)
    bm_sharpe   = _sharpe(bm_ann_ret, bm_ann_vol, risk_free_rate)
    bm_max_dd   = _max_drawdown(bm_ret)
    bm_tot_ret  = _total_return(bm_ret)

    # ── Cumulative return series for charts ───────────────────────────────────
    port_cumulative = _cumulative_returns(port_ret)
    bm_cumulative   = _cumulative_returns(bm_ret)

    period_years = len(port_ret) / TRADING_DAYS

    log.info(
        "Backtest complete — Portfolio: CAGR=%.2f%%, Vol=%.2f%%, Sharpe=%.3f, MaxDD=%.2f%%  |  "
        "Benchmark: CAGR=%.2f%%, Sharpe=%.3f",
        ann_ret * 100, ann_vol * 100, sharpe, max_dd * 100,
        bm_ann_ret * 100, bm_sharpe,
    )

    return BacktestResult(
        portfolio_cumulative        = port_cumulative,
        benchmark_cumulative        = bm_cumulative,
        portfolio_daily_returns     = port_ret,
        annualized_return           = ann_ret,
        annualized_volatility       = ann_vol,
        sharpe_ratio                = sharpe,
        max_drawdown                = max_dd,
        calmar_ratio                = calmar,
        total_return                = tot_ret,
        win_rate                    = win_r,
        benchmark_annualized_return = bm_ann_ret,
        benchmark_sharpe            = bm_sharpe,
        benchmark_max_drawdown      = bm_max_dd,
        benchmark_total_return      = bm_tot_ret,
        period_years                = period_years,
        start_date                  = port_ret.index[0].to_pydatetime(),
        end_date                    = port_ret.index[-1].to_pydatetime(),
        risk_free_rate              = risk_free_rate,
        weights                     = weights,
    )


def backtest_stats_table(result: BacktestResult) -> pd.DataFrame:
    """
    Return a tidy DataFrame comparing portfolio vs. benchmark performance.
    Ready to display directly in Streamlit with st.dataframe() or st.table().
    """
    def fmt_pct(v): return f"{v*100:.2f}%"
    def fmt_f(v):   return f"{v:.3f}"

    rows = {
        "Metric": [
            "Total Return",
            "Ann. Return (CAGR)",
            "Ann. Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
            "Calmar Ratio",
            "Win Rate",
        ],
        "Portfolio": [
            fmt_pct(result.total_return),
            fmt_pct(result.annualized_return),
            fmt_pct(result.annualized_volatility),
            fmt_f(result.sharpe_ratio),
            fmt_pct(result.max_drawdown),
            fmt_f(result.calmar_ratio),
            fmt_pct(result.win_rate),
        ],
        "NIFTY 50 Benchmark": [
            fmt_pct(result.benchmark_total_return),
            fmt_pct(result.benchmark_annualized_return),
            "—",
            fmt_f(result.benchmark_sharpe),
            fmt_pct(result.benchmark_max_drawdown),
            "—",
            "—",
        ],
    }
    return pd.DataFrame(rows)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data import get_market_data
    from optimize import get_efficient_frontier

    print("=" * 60)
    print("  Investigate App — Phase 3: Backtesting Validation")
    print("=" * 60)

    print("\n[1/3] Loading market data...")
    md = get_market_data(period_years=5)

    print("\n[2/3] Computing efficient frontier...")
    frontier = get_efficient_frontier(md)

    print("\n[3/3] Running backtest on each preset portfolio...")
    print()

    for name, portfolio in [
        ("Conservative", frontier.conservative),
        ("Balanced",     frontier.balanced),
        ("Aggressive",   frontier.aggressive),
        ("Min-Vol",      frontier.min_volatility),
        ("Max-Sharpe",   frontier.max_sharpe),
    ]:
        result = run_backtest(portfolio.weights, md, lookback_years=5)
        print(f"  ── {name} (5-year backtest) ──")
        print(f"     CAGR            : {result.annualized_return*100:+.2f}%")
        print(f"     Volatility      : {result.annualized_volatility*100:.2f}%")
        print(f"     Sharpe Ratio    : {result.sharpe_ratio:.3f}")
        print(f"     Max Drawdown    : {result.max_drawdown*100:.2f}%")
        print(f"     Total Return    : {result.total_return*100:+.2f}%")
        print(f"     vs. Benchmark   : {result.benchmark_annualized_return*100:+.2f}% CAGR")
        print()
