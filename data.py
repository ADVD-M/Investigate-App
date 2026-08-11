"""
data.py — Phase 1: Data Layer
==============================
Fetches and caches historical price data for Indian market assets using yfinance.
Computes returns and covariance matrices needed by the optimization engine.

Financial concepts:
- Daily returns: percentage change in price each day
- Annualized return: daily return scaled to ~252 trading days per year
- Covariance matrix: measures how assets move together; the basis of MPT risk calculation
"""

import os
import logging
from collections import namedtuple
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BENCHMARK_TICKER = "^NSEI"
CACHE_DIR        = "data_cache"
PRICES_FILE      = os.path.join(CACHE_DIR, "prices.csv")
BENCHMARK_FILE   = os.path.join(CACHE_DIR, "benchmark.csv")
TIMESTAMP_FILE   = os.path.join(CACHE_DIR, "last_updated.txt")
CACHE_TTL_HOURS  = 24
TRADING_DAYS     = 252
MAX_FILL_DAYS    = 5
MAX_MISSING_PCT  = 0.20

# ══════════════════════════════════════════════════════════════════════════════
# ASSET UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

# ── ETFs & Index Instruments ──────────────────────────────────────────────────
ETF_ASSETS: dict[str, str] = {
    # Broad index
    "NIFTYBEES.NS":  "Nifty 50 ETF",
    "JUNIORBEES.NS": "Nifty Next 50 ETF",
    "SETFNIF50.NS":  "SBI Nifty 50 ETF",
    # Sector ETFs
    "BANKBEES.NS":   "Bank Nifty ETF",
    "ITBEES.NS":     "IT Sector ETF",
    "CPSE.NS":       "CPSE ETF (PSU Basket)",
    "MOM100.NS":     "Midcap 100 ETF",
    # Commodity / Alternatives
    "GOLDBEES.NS":   "Gold ETF",
    "SILVERBEES.NS": "Silver ETF",
    # Debt / Money Market
    "LIQUIDBEES.NS": "Liquid / Debt ETF",
    # International
    "N100.NS":       "Nasdaq 100 ETF (US Tech)",
}

# ── Nifty 50 — core large-cap index ──────────────────────────────────────────
NIFTY50_SECTORS: dict[str, dict[str, str]] = {
    "Financial Services": {
        "HDFCBANK.NS":   "HDFC Bank",
        "ICICIBANK.NS":  "ICICI Bank",
        "KOTAKBANK.NS":  "Kotak Mahindra Bank",
        "AXISBANK.NS":   "Axis Bank",
        "SBIN.NS":       "State Bank of India",
        "BAJFINANCE.NS": "Bajaj Finance",
        "BAJAJFINSV.NS": "Bajaj Finserv",
        "INDUSINDBK.NS": "IndusInd Bank",
        "HDFCLIFE.NS":   "HDFC Life Insurance",
        "SBILIFE.NS":    "SBI Life Insurance",
        "SHRIRAMFIN.NS": "Shriram Finance",
    },
    "Information Technology": {
        "TCS.NS":     "Tata Consultancy Services",
        "INFY.NS":    "Infosys",
        "HCLTECH.NS": "HCL Technologies",
        "WIPRO.NS":   "Wipro",
        "TECHM.NS":   "Tech Mahindra",
    },
    "Energy & Utilities": {
        "RELIANCE.NS":  "Reliance Industries",
        "ONGC.NS":      "Oil & Natural Gas Corp",
        "BPCL.NS":      "Bharat Petroleum",
        "NTPC.NS":      "NTPC",
        "POWERGRID.NS": "Power Grid Corp",
        "COALINDIA.NS": "Coal India",
    },
    "Consumer Goods": {
        "HINDUNILVR.NS": "Hindustan Unilever",
        "ITC.NS":        "ITC",
        "NESTLEIND.NS":  "Nestle India",
        "BRITANNIA.NS":  "Britannia Industries",
        "TATACONSUM.NS": "Tata Consumer Products",
        "TITAN.NS":      "Titan Company",
        "TRENT.NS":      "Trent",
    },
    "Automobiles": {
        "MARUTI.NS":     "Maruti Suzuki",
        "TATAMOTORS.NS": "Tata Motors",
        "M&M.NS":        "Mahindra & Mahindra",
        "BAJAJ-AUTO.NS": "Bajaj Auto",
        "HEROMOTOCO.NS": "Hero MotoCorp",
        "EICHERMOT.NS":  "Eicher Motors",
    },
    "Pharma & Healthcare": {
        "SUNPHARMA.NS":  "Sun Pharmaceutical",
        "DRREDDY.NS":    "Dr. Reddy's Laboratories",
        "CIPLA.NS":      "Cipla",
        "APOLLOHOSP.NS": "Apollo Hospitals",
    },
    "Materials & Metals": {
        "TATASTEEL.NS":  "Tata Steel",
        "JSWSTEEL.NS":   "JSW Steel",
        "HINDALCO.NS":   "Hindalco Industries",
        "ULTRACEMCO.NS": "UltraTech Cement",
        "GRASIM.NS":     "Grasim Industries",
        "ASIANPAINT.NS": "Asian Paints",
    },
    "Industrials & Others": {
        "LT.NS":         "Larsen & Toubro",
        "ADANIENT.NS":   "Adani Enterprises",
        "ADANIPORTS.NS": "Adani Ports & SEZ",
        "BEL.NS":        "Bharat Electronics",
        "BHARTIARTL.NS": "Bharti Airtel",
        "ZOMATO.NS":     "Zomato",
    },
}

# ── Nifty Next 50 — next tier large-caps ─────────────────────────────────────
NIFTY_NEXT50_SECTORS: dict[str, dict[str, str]] = {
    "Financial Services (Next 50)": {
        "PNB.NS":        "Punjab National Bank",
        "BANKBARODA.NS": "Bank of Baroda",
        "IDFCFIRSTB.NS": "IDFC First Bank",
        "FEDERALBNK.NS": "Federal Bank",
        "CHOLAFIN.NS":   "Cholamandalam Investment",
        "MUTHOOTFIN.NS": "Muthoot Finance",
        "BAJAJHLDNG.NS": "Bajaj Holdings & Investment",
        "ICICIPRULI.NS": "ICICI Prudential Life",
        "SBICARD.NS":    "SBI Cards & Payment",
        "LICI.NS":       "LIC of India",
        "PNBHOUSING.NS": "PNB Housing Finance",
        "SUNDARMFIN.NS": "Sundaram Finance",
        "CANFINHOME.NS": "Can Fin Homes",
    },
    "IT & Tech (Next 50)": {
        "LTIM.NS":       "LTIMindtree",
        "LTTS.NS":       "L&T Technology Services",
        "MPHASIS.NS":    "Mphasis",
        "PERSISTENT.NS": "Persistent Systems",
        "TATAELXSI.NS":  "Tata Elxsi",
        "OFSS.NS":       "Oracle Financial Services",
        "COFORGE.NS":    "Coforge",
        "NAUKRI.NS":     "Info Edge (Naukri)",
        "KPITTECH.NS":   "KPIT Technologies",
        "TATATECH.NS":   "Tata Technologies",
    },
    "Energy & Utilities (Next 50)": {
        "GAIL.NS":       "GAIL India",
        "IOC.NS":        "Indian Oil Corporation",
        "PETRONET.NS":   "Petronet LNG",
        "JSWENERGY.NS":  "JSW Energy",
        "TATAPOWER.NS":  "Tata Power",
        "ADANIGREEN.NS": "Adani Green Energy",
        "ADANIPOWER.NS": "Adani Power",
        "TORNTPOWER.NS": "Torrent Power",
        "CESC.NS":       "CESC",
        "SUZLON.NS":     "Suzlon Energy",
    },
    "Consumer (Next 50)": {
        "DABUR.NS":      "Dabur India",
        "MARICO.NS":     "Marico",
        "GODREJCP.NS":   "Godrej Consumer Products",
        "COLPAL.NS":     "Colgate Palmolive India",
        "PIDILITIND.NS": "Pidilite Industries",
        "PAGEIND.NS":    "Page Industries",
        "VBL.NS":        "Varun Beverages",
        "UBL.NS":        "United Breweries",
        "TVSMOTOR.NS":   "TVS Motor Company",
        "BERGEPAINT.NS": "Berger Paints India",
        "GODREJPROP.NS": "Godrej Properties",
        "INDHOTEL.NS":   "Indian Hotels (Taj)",
        "PGHH.NS":       "P&G Hygiene & Health Care",
        "NYKAA.NS":      "FSN E-Commerce (Nykaa)",
        "EASEMYTRIP.NS": "Easy Trip Planners",
    },
    "Automobiles (Next 50)": {
        "MOTHERSON.NS":  "Motherson Sumi Wiring",
        "ESCORTS.NS":    "Escorts Kubota",
        "BALKRISIND.NS": "Balkrishna Industries",
        "UNOMINDA.NS":   "UNO Minda",
        "BOSCHLTD.NS":   "Bosch India",
        "CUMMINSIND.NS": "Cummins India",
        "EXIDEIND.NS":   "Exide Industries",
        "MRF.NS":        "MRF",
        "GABRIEL.NS":    "Gabriel India (Auto Ancillary)",
        "TMCV.NS":       "Tata Motors Commercial Vehicles",
        "TMPV.NS":       "Tata Motors Passenger Vehicles",
    },
    "Pharma (Next 50)": {
        "DIVISLAB.NS":   "Divi's Laboratories",
        "LUPIN.NS":      "Lupin",
        "AUROPHARMA.NS": "Aurobindo Pharma",
        "TORNTPHARM.NS": "Torrent Pharmaceuticals",
        "ALKEM.NS":      "Alkem Laboratories",
        "ZYDUSLIFE.NS":  "Zydus Lifesciences",
        "ABBOTINDIA.NS": "Abbott India",
        "MAXHEALTH.NS":  "Max Healthcare Institute",
        "STAR.NS":       "Star Health & Allied Insurance",
        "IPCA.NS":       "Ipca Laboratories",
        "MANKIND.NS":    "Mankind Pharma",
    },
    "Materials & Industrials (Next 50)": {
        "AMBUJACEM.NS":  "Ambuja Cements",
        "SAIL.NS":       "Steel Authority of India",
        "VEDL.NS":       "Vedanta",
        "NMDC.NS":       "NMDC",
        "JSWINFRA.NS":   "JSW Infrastructure",
        "DALBHARAT.NS":  "Dalmia Bharat",
        "RAMCOCEM.NS":   "Ramco Cements",
        "SRF.NS":        "SRF Limited",
        "DEEPAKNTR.NS":  "Deepak Nitrite",
        "POLYCAB.NS":    "Polycab India",
        "HAVELLS.NS":    "Havells India",
        "SIEMENS.NS":    "Siemens India",
        "ABB.NS":        "ABB India",
        "SUPREMEIND.NS": "Supreme Industries",
        "SCHAEFFLER.NS": "Schaeffler India",
        "TIINDIA.NS":    "Tube Investments of India",
        "CGPOWER.NS":    "CG Power & Industrial",
        "HINDCOPPER.NS": "Hindustan Copper",
        "SILVERCASE.NS": "Silver Industries / Silvercase",
    },
    "Infrastructure & Others (Next 50)": {
        "DLF.NS":        "DLF",
        "INDUSTOWER.NS": "Indus Towers",
        "IRCTC.NS":      "IRCTC",
        "RECLTD.NS":     "REC Limited",
        "PFC.NS":        "Power Finance Corporation",
        "NHPC.NS":       "NHPC",
        "IRFC.NS":       "Indian Railway Finance Corp",
        "HAL.NS":        "Hindustan Aeronautics",
        "BHEL.NS":       "Bharat Heavy Electricals",
        "CONCOR.NS":     "Container Corp of India",
        "OBEROIRLTY.NS": "Oberoi Realty",
        "PHOENIXLTD.NS": "Phoenix Mills",
        "LODHA.NS":      "Macrotech Developers (Lodha)",
        "TATACOMM.NS":   "Tata Communications",
        "MFSL.NS":       "Max Financial Services",
        "ZEEL.NS":       "Zee Entertainment",
        "SUNTV.NS":      "Sun TV Network",
        "POLICYBZR.NS":  "PB Fintech (PolicyBazaar)",
        "PAYTM.NS":      "One97 Communications (Paytm)",
    },
}

# ── Flat lookups ──────────────────────────────────────────────────────────────

NIFTY50_STOCKS: dict[str, str] = {
    ticker: label
    for sector_stocks in NIFTY50_SECTORS.values()
    for ticker, label in sector_stocks.items()
}

NIFTY_NEXT50_STOCKS: dict[str, str] = {
    ticker: label
    for sector_stocks in NIFTY_NEXT50_SECTORS.values()
    for ticker, label in sector_stocks.items()
}

# Full asset universe available in the app
ALL_ASSETS: dict[str, str] = {
    **ETF_ASSETS,
    **NIFTY50_STOCKS,
    **NIFTY_NEXT50_STOCKS,
}

# Curated 10-asset default for quick first run (diverse across asset classes)
DEFAULT_SELECTION: list[str] = [
    "NIFTYBEES.NS",   # Broad equity index
    "GOLDBEES.NS",    # Gold / inflation hedge
    "LIQUIDBEES.NS",  # Debt proxy
    "HDFCBANK.NS",    # Financials
    "RELIANCE.NS",    # Energy / conglomerate
    "TCS.NS",         # IT
    "INFY.NS",        # IT
    "SUNPHARMA.NS",   # Pharma
    "HINDUNILVR.NS",  # Consumer staples
    "LT.NS",          # Industrials
]

# Legacy alias — kept for backward compatibility with tests
DEFAULT_ASSETS: dict[str, str] = {
    "NIFTYBEES.NS":  "Nifty 50 ETF (Equity)",
    "RELIANCE.NS":   "Reliance Industries (Equity)",
    "TCS.NS":        "Tata Consultancy Services (Equity)",
    "HDFCBANK.NS":   "HDFC Bank (Equity)",
    "GOLDBEES.NS":   "Gold ETF (Commodity)",
    "LIQUIDBEES.NS": "Liquid ETF (Debt Proxy)",
}

# Named tuple returned by get_market_data
MarketData = namedtuple(
    "MarketData",
    [
        "prices",
        "daily_returns",
        "annualized_returns",
        "covariance_matrix",
        "benchmark_prices",
        "benchmark_returns",
        "tickers",
        "period_years",
        "fetched_at",
    ],
)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_is_fresh() -> bool:
    if not os.path.exists(TIMESTAMP_FILE):
        return False
    try:
        with open(TIMESTAMP_FILE) as f:
            ts = datetime.fromisoformat(f.read().strip())
        return datetime.now() - ts < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False


def _write_timestamp() -> None:
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(datetime.now().isoformat())


def _read_timestamp() -> datetime:
    with open(TIMESTAMP_FILE) as f:
        return datetime.fromisoformat(f.read().strip())


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _fetch_prices(tickers: list[str], period_years: int) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    yfinance uses '.NS' suffix for NSE-listed securities.
    """
    end   = datetime.today()
    start = end - timedelta(days=period_years * 365)

    log.info("Fetching %d tickers from Yahoo Finance (%d-year window)...", len(tickers), period_years)

    raw = yf.download(
        tickers,
        start       = start.strftime("%Y-%m-%d"),
        end         = end.strftime("%Y-%m-%d"),
        auto_adjust = True,
        progress    = False,
        threads     = True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return prices


def _fetch_benchmark(period_years: int) -> pd.Series:
    """Download NIFTY 50 index prices for benchmark comparison."""
    end   = datetime.today()
    start = end - timedelta(days=period_years * 365)

    log.info("Fetching benchmark (%s)...", BENCHMARK_TICKER)
    raw = yf.download(
        BENCHMARK_TICKER,
        start       = start.strftime("%Y-%m-%d"),
        end         = end.strftime("%Y-%m-%d"),
        auto_adjust = True,
        progress    = False,
    )
    return raw["Close"].squeeze().rename(BENCHMARK_TICKER)


# ── Data cleaning ─────────────────────────────────────────────────────────────

def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data:
    1. Forward-fill short gaps (weekends, holidays) up to MAX_FILL_DAYS
    2. Drop assets with > MAX_MISSING_PCT missing values (e.g. recent IPOs)
    3. Drop rows where all assets are NaN
    """
    prices = prices.ffill(limit=MAX_FILL_DAYS)

    missing_frac = prices.isna().mean()
    bad_tickers  = missing_frac[missing_frac > MAX_MISSING_PCT].index.tolist()
    if bad_tickers:
        log.warning("Dropping %d asset(s) with >%.0f%% missing data: %s",
                    len(bad_tickers), MAX_MISSING_PCT * 100, bad_tickers)
        prices = prices.drop(columns=bad_tickers)

    prices = prices.dropna(how="all").dropna()
    return prices


# ── Return & covariance computation ──────────────────────────────────────────

def _compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute:
    - Daily returns:      r_t = (P_t − P_{t-1}) / P_{t-1}
    - Annualized return:  mean(daily_r) × 252
    - Annualized cov:     cov(daily_r)  × 252  (scales linearly under MPT)
    """
    daily_returns      = prices.pct_change().dropna()
    annualized_returns = daily_returns.mean() * TRADING_DAYS
    annualized_cov     = daily_returns.cov()  * TRADING_DAYS
    return daily_returns, annualized_returns, annualized_cov


# ── Public API ────────────────────────────────────────────────────────────────

def get_market_data(
    tickers: list[str] | None = None,
    period_years: int = 10,
    force_refresh: bool = False,
) -> MarketData:
    """
    Main entry point for the data layer.

    Parameters
    ----------
    tickers       : NSE tickers to fetch. Defaults to DEFAULT_SELECTION.
    period_years  : Lookback window in years.
    force_refresh : Bypass cache and re-fetch from Yahoo Finance.

    Note: Assets with > 20% missing data are automatically dropped (e.g. recent IPOs).
    """
    if tickers is None:
        tickers = DEFAULT_SELECTION

    _ensure_cache_dir()

    use_cache = (not force_refresh) and _cache_is_fresh() and os.path.exists(PRICES_FILE)

    if use_cache:
        log.info("Loading from cache (last updated: %s)...", _read_timestamp())
        prices          = pd.read_csv(PRICES_FILE,    index_col=0, parse_dates=True)
        benchmark_price = pd.read_csv(BENCHMARK_FILE, index_col=0, parse_dates=True).squeeze()

        available = [t for t in tickers if t in prices.columns]
        missing   = [t for t in tickers if t not in prices.columns]
        if missing:
            log.warning("Cache missing tickers %s — triggering refetch.", missing)
            use_cache = False
        else:
            prices     = prices[available]
            fetched_at = _read_timestamp()

    if not use_cache:
        raw_prices      = _fetch_prices(tickers, period_years)
        benchmark_price = _fetch_benchmark(period_years)

        prices = _clean_prices(raw_prices)
        benchmark_price = benchmark_price.reindex(prices.index).ffill().dropna()

        prices.to_csv(PRICES_FILE)
        benchmark_price.to_frame().to_csv(BENCHMARK_FILE)
        _write_timestamp()
        fetched_at = datetime.now()
        log.info("Data cached to '%s'.", CACHE_DIR)

    daily_returns, annualized_returns, annualized_cov = _compute_returns(prices)
    benchmark_returns = benchmark_price.pct_change().dropna()

    log.info(
        "Market data ready: %d assets, %d trading days (%.1f years)",
        len(prices.columns), len(prices), len(prices) / TRADING_DAYS,
    )

    return MarketData(
        prices             = prices,
        daily_returns      = daily_returns,
        annualized_returns = annualized_returns,
        covariance_matrix  = annualized_cov,
        benchmark_prices   = benchmark_price,
        benchmark_returns  = benchmark_returns,
        tickers            = list(prices.columns),
        period_years       = period_years,
        fetched_at         = fetched_at,
    )


def get_asset_labels(tickers: list[str] | None = None) -> dict[str, str]:
    """Return human-readable labels for the given tickers (from the full universe)."""
    if tickers is None:
        return ALL_ASSETS.copy()
    return {t: ALL_ASSETS.get(t, t) for t in tickers}


def get_sector_for_ticker(ticker: str) -> str:
    """Return the sector name for a given ticker, or 'ETF / Other'."""
    for sector, stocks in NIFTY50_SECTORS.items():
        if ticker in stocks:
            return sector
    for sector, stocks in NIFTY_NEXT50_SECTORS.items():
        if ticker in stocks:
            return f"{sector}"
    if ticker in ETF_ASSETS:
        return "ETF / Index"
    return "Other"


# ── Standalone validation ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Investigate App — Phase 1: Data Layer Validation")
    print("=" * 60)

    n50   = len(NIFTY50_STOCKS)
    nn50  = len(NIFTY_NEXT50_STOCKS)
    netfs = len(ETF_ASSETS)
    total = len(ALL_ASSETS)

    print(f"\n✓ Asset universe size:")
    print(f"  ETFs           : {netfs}")
    print(f"  Nifty 50       : {n50}")
    print(f"  Nifty Next 50+ : {nn50}")
    print(f"  ─────────────────")
    print(f"  TOTAL          : {total}")

    print("\n[Fetching default selection...]")
    md = get_market_data(period_years=5)

    print(f"\n✓ Tickers loaded    : {md.tickers}")
    print(f"✓ Date range        : {md.prices.index[0].date()} → {md.prices.index[-1].date()}")
    print(f"✓ Trading days      : {len(md.prices)}")

    print("\n── Annualized Returns ──────────────────────────────────")
    for ticker, ret in md.annualized_returns.items():
        label = ALL_ASSETS.get(ticker, ticker)
        print(f"  {ticker:<20} {ret*100:+.2f}%   ({label})")
