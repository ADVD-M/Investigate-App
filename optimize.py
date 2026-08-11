"""
optimize.py — Phase 2: Optimization Engine
===========================================
Computes the Efficient Frontier and key portfolios using Modern Portfolio Theory (MPT).
Uses PyPortfolioOpt as the optimization backend.

Financial concepts:
- Efficient Frontier: The set of portfolios that offer the highest expected return
  for a given level of risk (volatility). No rational investor would choose a
  portfolio that lies below this curve.
- Minimum Volatility portfolio: The single point on the frontier with the lowest
  possible risk. Good for conservative investors.
- Maximum Sharpe Ratio portfolio: The portfolio with the best risk-adjusted return.
  The Sharpe ratio = (return - risk_free_rate) / volatility.
- Weights: The fraction of the total portfolio allocated to each asset (sum = 1.0).
"""

import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# India's approximate risk-free rate (91-day T-bill yield, ~6.5% as of 2024)
DEFAULT_RISK_FREE_RATE = 0.065

# Number of portfolios to compute along the efficient frontier curve
# More points = smoother curve; 50 is a good balance of speed vs. resolution
DEFAULT_N_POINTS = 50

# Named tuple for a single portfolio's result
PortfolioResult = namedtuple(
    "PortfolioResult",
    [
        "weights",           # dict: {ticker: weight}, weights sum to 1.0
        "expected_return",   # float: annualized expected return
        "volatility",        # float: annualized volatility (std dev of returns)
        "sharpe_ratio",      # float: risk-adjusted return measure
        "label",             # str:   human-readable name for this portfolio
    ],
)

# Full frontier result — everything the UI and backtest layer need
FrontierResult = namedtuple(
    "FrontierResult",
    [
        "frontier_points",   # list[PortfolioResult]: all points on the efficient frontier
        "min_volatility",    # PortfolioResult: minimum risk portfolio
        "max_sharpe",        # PortfolioResult: maximum Sharpe ratio portfolio
        "conservative",      # PortfolioResult: low-risk preset (~20th percentile of frontier)
        "balanced",          # PortfolioResult: mid-risk preset (~50th percentile)
        "aggressive",        # PortfolioResult: high-return preset (~80th percentile)
        "tickers",           # list[str]: asset tickers used
        "risk_free_rate",    # float: risk-free rate used in Sharpe computation
    ],
)


# ── Core helpers ──────────────────────────────────────────────────────────────

def _build_ef(mu: pd.Series, S: pd.DataFrame, weight_bounds=(0, 1)) -> EfficientFrontier:
    """
    Construct a PyPortfolioOpt EfficientFrontier object.

    Parameters
    ----------
    mu            : Annualized expected returns (pd.Series)
    S             : Annualized covariance matrix (pd.DataFrame)
    weight_bounds : (min_weight, max_weight) per asset.
                    (0, 1) = long-only; no shorting (appropriate for retail investors)
    """
    return EfficientFrontier(mu, S, weight_bounds=weight_bounds)


def _ef_to_result(ef: EfficientFrontier, label: str, risk_free_rate: float) -> PortfolioResult:
    """
    Extract clean weights and performance metrics from a solved EfficientFrontier.
    Must be called AFTER calling ef.min_volatility(), ef.max_sharpe(), etc.
    """
    weights = ef.clean_weights()  # rounds tiny weights to 0 to avoid noise
    perf    = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
    # perf = (expected_return, volatility, sharpe_ratio)
    return PortfolioResult(
        weights         = dict(weights),
        expected_return = perf[0],
        volatility      = perf[1],
        sharpe_ratio    = perf[2],
        label           = label,
    )


def _sweep_frontier(
    mu: pd.Series,
    S: pd.DataFrame,
    risk_free_rate: float,
    n_points: int,
) -> list[PortfolioResult]:
    """
    Sweep the efficient frontier by solving for the minimum-variance portfolio
    at each of n_points target return levels between the global min-vol return
    and the maximum achievable return.

    This gives the coordinates of the frontier curve for the Plotly chart.
    """
    # Find the min and max feasible returns for the sweep range
    ef_min = _build_ef(mu, S)
    ef_min.min_volatility()
    min_ret, _, _ = ef_min.portfolio_performance(verbose=False)

    max_ret = float(mu.max())  # can't exceed highest individual asset return

    # Clip the range slightly inward to avoid numerical issues at the boundaries
    targets = np.linspace(min_ret * 1.001, max_ret * 0.999, n_points)

    frontier_points = []
    for i, target_return in enumerate(targets):
        try:
            ef = _build_ef(mu, S)
            ef.efficient_return(target_return)
            result = _ef_to_result(ef, label=f"Frontier point {i+1}", risk_free_rate=risk_free_rate)
            frontier_points.append(result)
        except Exception as exc:
            # Some target returns may be infeasible (e.g. above the feasible region)
            log.debug("Skipping frontier point at target return %.4f: %s", target_return, exc)

    log.info("Computed %d / %d efficient frontier points.", len(frontier_points), n_points)
    return frontier_points


def _pick_preset(frontier_points: list[PortfolioResult], percentile: float) -> PortfolioResult:
    """
    Select a frontier point at the given percentile of the volatility range.
    percentile=0.2 → conservative (near min-vol end)
    percentile=0.5 → balanced (midpoint)
    percentile=0.8 → aggressive (near max-return end)
    """
    idx = int(round(percentile * (len(frontier_points) - 1)))
    idx = max(0, min(idx, len(frontier_points) - 1))
    return frontier_points[idx]


# ── Public API ────────────────────────────────────────────────────────────────

def get_efficient_frontier(
    market_data,                          # MarketData namedtuple from data.py
    n_points: int = DEFAULT_N_POINTS,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> FrontierResult:
    """
    Compute the efficient frontier for the given market data.

    Frontier curve:  unconstrained (0, 1) — shows the full theoretical envelope.
    Named presets:   each solved with its own diversification caps so outputs
                     are actionable for real investors:

        Conservative  → min_volatility,       max 20% per stock
        Balanced      → max_sharpe,            max 25% per stock
        Aggressive    → efficient_return(80%), max 40% per stock
        Min Volatility → min_volatility,       unconstrained (mathematical)
        Max Sharpe    → max_sharpe,            unconstrained (theoretical reference)

    The caps scale up automatically when fewer than 5/4/3 assets are selected
    so the optimisation is always feasible.
    """
    mu = market_data.annualized_returns
    S  = market_data.covariance_matrix
    n  = len(mu)

    log.info(
        "Computing efficient frontier for %d assets (risk-free rate: %.2f%%)...",
        n, risk_free_rate * 100
    )

    # ── Adaptive per-stock caps ───────────────────────────────────────────────
    # max(hard_cap, 1/n) guarantees a feasible allocation even when the user
    # has selected only 2–4 stocks (e.g. with 3 stocks, conservative cap = 33%).
    _cap_con = max(0.20, 1.0 / n)   # Conservative: ideally ≥5 stocks
    _cap_bal = max(0.25, 1.0 / n)   # Balanced:     ideally ≥4 stocks
    _cap_agg = max(0.40, 1.0 / n)   # Aggressive:   ideally ≥3 stocks

    # ── Frontier sweep (UNCONSTRAINED) — for the chart curve ─────────────────
    # Stays unconstrained so the chart always shows the true theoretical frontier.
    frontier_points = _sweep_frontier(mu, S, risk_free_rate, n_points)

    # ── Min Volatility (unconstrained — mathematical minimum risk) ────────────
    ef_minvol = _build_ef(mu, S)
    ef_minvol.min_volatility()
    min_vol_result = _ef_to_result(ef_minvol, "Minimum Volatility", risk_free_rate)
    log.info(
        "Min-Vol: return=%.2f%%, vol=%.2f%%, Sharpe=%.3f",
        min_vol_result.expected_return * 100,
        min_vol_result.volatility * 100,
        min_vol_result.sharpe_ratio,
    )

    # ── Max Sharpe (unconstrained — theoretical reference) ────────────────────
    ef_maxsh = _build_ef(mu, S)
    ef_maxsh.max_sharpe(risk_free_rate=risk_free_rate)
    max_sharpe_result = _ef_to_result(ef_maxsh, "Maximum Sharpe Ratio", risk_free_rate)
    log.info(
        "Max-Sharpe: return=%.2f%%, vol=%.2f%%, Sharpe=%.3f",
        max_sharpe_result.expected_return * 100,
        max_sharpe_result.volatility * 100,
        max_sharpe_result.sharpe_ratio,
    )

    # ── Conservative (constrained: minimise volatility, max 20% per stock) ───
    ef_con = _build_ef(mu, S, weight_bounds=(0, _cap_con))
    ef_con.min_volatility()
    conservative = _ef_to_result(ef_con, "Conservative", risk_free_rate)
    log.info(
        "Conservative (cap=%.0f%%): return=%.2f%%, vol=%.2f%%, Sharpe=%.3f",
        _cap_con * 100,
        conservative.expected_return * 100,
        conservative.volatility * 100,
        conservative.sharpe_ratio,
    )

    # ── Balanced (constrained: maximise Sharpe, max 25% per stock) ───────────
    ef_bal = _build_ef(mu, S, weight_bounds=(0, _cap_bal))
    ef_bal.max_sharpe(risk_free_rate=risk_free_rate)
    balanced = _ef_to_result(ef_bal, "Balanced", risk_free_rate)
    log.info(
        "Balanced (cap=%.0f%%): return=%.2f%%, vol=%.2f%%, Sharpe=%.3f",
        _cap_bal * 100,
        balanced.expected_return * 100,
        balanced.volatility * 100,
        balanced.sharpe_ratio,
    )

    # ── Aggressive (constrained: 80th-percentile return, max 40% per stock) ──
    # Target return = 80th percentile of the unconstrained frontier.
    # Falls back to max_sharpe (constrained) if that return is infeasible.
    if frontier_points:
        _agg_idx    = int(round(0.80 * (len(frontier_points) - 1)))
        _agg_target = frontier_points[min(_agg_idx, len(frontier_points) - 1)].expected_return
    else:
        _agg_target = float(mu.max()) * 0.80

    ef_agg = _build_ef(mu, S, weight_bounds=(0, _cap_agg))
    try:
        ef_agg.efficient_return(_agg_target)
        aggressive = _ef_to_result(ef_agg, "Aggressive", risk_free_rate)
    except Exception:
        log.debug("Aggressive efficient_return infeasible under constraints; falling back to max_sharpe.")
        ef_agg = _build_ef(mu, S, weight_bounds=(0, _cap_agg))
        ef_agg.max_sharpe(risk_free_rate=risk_free_rate)
        aggressive = _ef_to_result(ef_agg, "Aggressive", risk_free_rate)

    log.info(
        "Aggressive (cap=%.0f%%): return=%.2f%%, vol=%.2f%%, Sharpe=%.3f",
        _cap_agg * 100,
        aggressive.expected_return * 100,
        aggressive.volatility * 100,
        aggressive.sharpe_ratio,
    )

    return FrontierResult(
        frontier_points = frontier_points,
        min_volatility  = min_vol_result,
        max_sharpe      = max_sharpe_result,
        conservative    = conservative,
        balanced        = balanced,
        aggressive      = aggressive,
        tickers         = list(mu.index),
        risk_free_rate  = risk_free_rate,
    )


def frontier_to_dataframe(frontier: FrontierResult) -> pd.DataFrame:
    """
    Convert the list of frontier points into a tidy DataFrame for easy plotting.

    Columns: volatility, expected_return, sharpe_ratio, label, + one column per ticker (weights)
    """
    rows = []
    for pt in frontier.frontier_points:
        row = {
            "volatility":       pt.volatility,
            "expected_return":  pt.expected_return,
            "sharpe_ratio":     pt.sharpe_ratio,
            "label":            pt.label,
        }
        row.update(pt.weights)
        rows.append(row)

    return pd.DataFrame(rows)


def weights_to_series(portfolio: PortfolioResult, ticker_labels: dict | None = None) -> pd.Series:
    """
    Convert a portfolio's weights dict to a named pd.Series, optionally
    mapping tickers to human-readable labels (for pie chart display).
    """
    weights = {
        (ticker_labels.get(t, t) if ticker_labels else t): w
        for t, w in portfolio.weights.items()
        if w > 0.001  # skip negligible weights
    }
    return pd.Series(weights, name=portfolio.label)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import get_market_data, get_asset_labels, DEFAULT_ASSETS

    print("=" * 60)
    print("  Investigate App — Phase 2: Optimization Engine Validation")
    print("=" * 60)

    # Load data (will use cache if available)
    print("\n[1/3] Loading market data...")
    md = get_market_data(period_years=5)

    # Compute frontier
    print("\n[2/3] Computing efficient frontier...")
    frontier = get_efficient_frontier(md)

    labels = get_asset_labels(md.tickers)

    print(f"\n✓ Frontier points computed : {len(frontier.frontier_points)}")
    print(f"✓ Tickers used             : {frontier.tickers}")
    print(f"✓ Risk-free rate           : {frontier.risk_free_rate*100:.2f}%")

    # Print key portfolios
    print("\n[3/3] Key Portfolios:")
    print("─" * 60)
    for name, p in [
        ("Minimum Volatility", frontier.min_volatility),
        ("Maximum Sharpe",     frontier.max_sharpe),
        ("Conservative",       frontier.conservative),
        ("Balanced",           frontier.balanced),
        ("Aggressive",         frontier.aggressive),
    ]:
        print(f"\n  ── {name} ──")
        print(f"     Expected Return : {p.expected_return*100:+.2f}%")
        print(f"     Volatility      : {p.volatility*100:.2f}%")
        print(f"     Sharpe Ratio    : {p.sharpe_ratio:.3f}")
        print("     Weights:")
        ws = weights_to_series(p, labels)
        for asset, w in ws.items():
            print(f"       {asset:<40} {w*100:.1f}%")

    print("\n── Frontier DataFrame (first 5 rows) ──────────────────")
    df = frontier_to_dataframe(frontier)
    print(df[["volatility", "expected_return", "sharpe_ratio"]].head().to_string(index=False))

    # Sanity check: all weights sum to ~1.0
    print("\n── Sanity Checks ───────────────────────────────────────")
    for name, p in [
        ("Min-Vol",      frontier.min_volatility),
        ("Max-Sharpe",   frontier.max_sharpe),
        ("Conservative", frontier.conservative),
        ("Balanced",     frontier.balanced),
        ("Aggressive",   frontier.aggressive),
    ]:
        total = sum(p.weights.values())
        status = "✓" if abs(total - 1.0) < 0.01 else "✗"
        print(f"  {status} {name:<20} weights sum = {total:.6f}")
