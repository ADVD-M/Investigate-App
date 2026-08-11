"""
app.py — Phase 4: Streamlit Frontend (Full Revamp)
====================================================
Investigate — Frontier Portfolio Engine
India · NSE Markets · Modern Portfolio Theory

⚠️  DISCLAIMER: Educational tool only. Not financial advice.
    Past performance does not guarantee future results.
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import uuid

from data     import (
    get_market_data,
    ALL_ASSETS, DEFAULT_SELECTION,
)
from optimize import get_efficient_frontier, frontier_to_dataframe, weights_to_series
from backtest import run_backtest, backtest_stats_table
import db as _db

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Investigate — Frontier Portfolio Engine",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Persistence: initialise DB + resolve user identity ───────────────────────

_db.init_db()

# Get or create a UUID for this visitor from the URL (?uid=...)
# The UUID is the user's anonymous identity. They keep their portfolio by
# bookmarking their personal URL. No login required.
if "uid" not in st.query_params:
    st.query_params["uid"] = str(uuid.uuid4())

_USER_ID: str = st.query_params["uid"]

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ───────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
  background: #080808;
  background-image: radial-gradient(ellipse 70% 40% at 50% -5%,
    rgba(0,208,132,0.05) 0%, transparent 65%);
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #0a0a0a !important;
  border-right: 1px solid rgba(0,208,132,0.12) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p { color: #9e9e9e !important; font-size: 0.82rem; }

/* ── Headings ────────────────────────────────────────────── */
h1, h2, h3 { letter-spacing: -0.02em !important; }

/* ── Hero ────────────────────────────────────────────────── */
.hero-brand {
  font-size: 2.8rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  background: linear-gradient(135deg, #ffffff 0%, #00d084 55%, #69ff47 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
  margin: 0;
}
.hero-sub {
  font-size: 0.85rem;
  color: #2e2e2e;
  font-weight: 400;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 0.45rem;
}

/* ── Section headers ─────────────────────────────────────── */
.sec-header {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: #00d084;
  margin: 2rem 0 1rem 0;
}
.sec-header::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(0,208,132,0.25), transparent);
}

/* ── Status bar ──────────────────────────────────────────── */
.status-bar {
  background: rgba(0,208,132,0.025);
  border: 1px solid rgba(0,208,132,0.1);
  border-radius: 6px;
  padding: 0.55rem 1rem;
  font-size: 0.76rem;
  color: #333;
  font-family: 'JetBrains Mono', monospace;
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}
.status-bar span { color: #00d084; font-weight: 500; }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
  background: rgba(0,208,132,0.04) !important;
  border: 1px solid rgba(0,208,132,0.2) !important;
  color: #9e9e9e !important;
  border-radius: 5px !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
  padding: 0.35rem 0.65rem !important;
  white-space: nowrap !important;
  line-height: 1.4 !important;
  min-height: 0 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}
.stButton > button:hover {
  background: rgba(0,208,132,0.1) !important;
  border-color: #00d084 !important;
  color: #00d084 !important;
  box-shadow: 0 2px 10px rgba(0,208,132,0.1) !important;
}

/* ── Text inputs — highlighted border ───────────────────── */
[data-testid="stTextInput"] > div > div > input {
  background: #0d0d0d !important;
  border: 1.5px solid rgba(0,208,132,0.35) !important;
  border-radius: 5px !important;
  color: #f0f0f0 !important;
  font-size: 0.83rem !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
  padding: 0.45rem 0.75rem !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
  border-color: #00d084 !important;
  box-shadow: 0 0 0 2px rgba(0,208,132,0.1) !important;
  outline: none !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder {
  color: #2a2a2a !important;
}

/* ── Multiselect ─────────────────────────────────────────── */
[data-testid="stMultiSelect"] > div > div {
  background: #0d0d0d !important;
  border: 1.5px solid rgba(255,255,255,0.08) !important;
  border-radius: 5px !important;
  transition: border-color 0.15s !important;
}
[data-testid="stMultiSelect"] > div > div:focus-within {
  border-color: rgba(0,208,132,0.3) !important;
}
[data-baseweb="tag"] {
  background: rgba(0,208,132,0.1) !important;
  border: 1px solid rgba(0,208,132,0.25) !important;
  border-radius: 3px !important;
}
[data-baseweb="tag"] span { color: #00d084 !important; }

/* ── Metric card ─────────────────────────────────────────── */
.mcard {
  background: #0d0d0d;
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 8px;
  padding: 0.85rem 1rem;
  text-align: center;
  transition: border-color 0.15s, transform 0.15s;
  position: relative;
  overflow: hidden;
}
.mcard::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(0,208,132,0.35), transparent);
}
.mcard:hover {
  border-color: rgba(0,208,132,0.18);
  transform: translateY(-1px);
}
.mcard .lbl {
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: #333;
  font-weight: 600;
  margin-bottom: 0.3rem;
}
.mcard .val {
  font-size: 1.35rem;
  font-weight: 700;
  color: #e0e0e0;
  font-family: 'JetBrains Mono', monospace;
}
.mcard .val.pos { color: #00d084; }
.mcard .val.neg { color: #e53935; }
.mcard .val.info { color: #9e9e9e; }

/* ── Disclaimer ──────────────────────────────────────────── */
.disc {
  background: rgba(229,57,53,0.04);
  border: 1px solid rgba(229,57,53,0.12);
  border-radius: 6px;
  padding: 0.6rem 0.9rem;
  font-size: 0.72rem;
  color: #5a3a3a;
  line-height: 1.55;
}
.disc strong { color: #ef9a9a; }

/* ── How-to / concept cards ──────────────────────────────── */
.how-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin-top: 0.75rem;
}
.how-card {
  background: #0d0d0d;
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 10px;
  padding: 1.1rem;
  transition: border-color 0.15s, transform 0.15s;
}
.how-card:hover {
  border-color: rgba(0,208,132,0.18);
  transform: translateY(-2px);
}
.how-card .step-num {
  font-size: 1.8rem;
  font-weight: 800;
  font-family: 'JetBrains Mono', monospace;
  color: rgba(0,208,132,0.18);
  line-height: 1;
  margin-bottom: 0.35rem;
}
.how-card .step-title { font-size: 0.84rem; font-weight: 600; color: #e0e0e0; margin-bottom: 0.3rem; }
.how-card .step-body  { font-size: 0.77rem; color: #333; line-height: 1.6; }

.concept-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.6rem;
  margin-top: 0.6rem;
}
.concept-card {
  background: rgba(0,208,132,0.025);
  border: 1px solid rgba(0,208,132,0.1);
  border-radius: 8px;
  padding: 0.85rem 1rem;
}
.concept-card .c-title { font-size: 0.77rem; font-weight: 600; color: #00d084; margin-bottom: 0.3rem; }
.concept-card .c-body  { font-size: 0.74rem; color: #333; line-height: 1.55; }

.footer {
  text-align: center;
  font-size: 0.7rem;
  color: #181818;
  padding: 2rem 0 1rem 0;
  border-top: 1px solid rgba(255,255,255,0.03);
  margin-top: 2rem;
}

hr { border-color: rgba(255,255,255,0.04) !important; margin: 0.5rem 0 !important; }

[data-testid="stPlotlyChart"] > div {
  border-radius: 10px !important;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)



# ── Plotly base theme ─────────────────────────────────────────────────────────

_BASE_LAYOUT = dict(
    paper_bgcolor = "rgba(8,8,8,0)",
    plot_bgcolor  = "rgba(10,10,10,0.6)",
    font          = dict(family="Inter, sans-serif", color="#555", size=12),
    xaxis         = dict(
        gridcolor      = "rgba(255,255,255,0.03)",
        zerolinecolor  = "rgba(0,208,132,0.12)",
        tickfont       = dict(size=11, color="#555"),
    ),
    yaxis         = dict(
        gridcolor      = "rgba(255,255,255,0.03)",
        zerolinecolor  = "rgba(0,208,132,0.12)",
        tickfont       = dict(size=11, color="#555"),
    ),
    legend        = dict(
        bgcolor      = "rgba(8,8,8,0.9)",
        bordercolor  = "rgba(0,208,132,0.1)",
        borderwidth  = 1,
        font         = dict(size=11, color="#888"),
    ),
    hoverlabel    = dict(bgcolor="#0d0d0d", bordercolor="#00d084", font_color="#f0f0f0", font_size=12),
    margin        = dict(l=55, r=25, t=45, b=50),
)

def _layout(**overrides) -> dict:
    """Merge base layout with per-chart overrides, avoiding duplicate-key errors."""
    return {**_BASE_LAYOUT, **overrides}


# ── Palette  (financial terminal: black / green / red / white / grey) ──────────
ACCENT     = "#00d084"   # Financial green — primary accent
SUCCESS    = "#00d084"   # Gain green
DANGER     = "#e53935"   # Loss red
NEUTRAL    = "#9e9e9e"   # Mid grey
GOLD       = "#ffd54f"   # Yellow-gold for Max Sharpe
PURPLE     = "#9e9e9e"   # Grey replaces purple (cleaner look)
PIE_COLORS = [ACCENT, "#007a4d", DANGER, GOLD, "#78909c", "#e65100", "#1565c0", "#558b2f", "#f06292", "#80cbc4"]

PRESET_META = {
    "conservative"  : ("Conservative",  "Low risk, capital preservation"),
    "balanced"      : ("Balanced",       "Moderate risk & return"),
    "aggressive"    : ("Aggressive",     "High risk, high potential"),
    "min_volatility": ("Min Volatility", "Mathematically least-risky portfolio"),
    "max_sharpe"    : ("Max Sharpe",     "Best risk-adjusted return (Sharpe)"),
}

POINT_COLORS = {
    "min_volatility": NEUTRAL,
    "max_sharpe"    : GOLD,
    "conservative"  : SUCCESS,
    "balanced"      : "#ffffff",
    "aggressive"    : DANGER,
}


# ── Cached functions ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching data from Yahoo Finance / NSE…")
def load_data(tickers: tuple[str, ...], period_years: int, force: bool) -> object:
    return get_market_data(list(tickers), period_years=period_years, force_refresh=force)


@st.cache_data(show_spinner="Computing efficient frontier…")
def compute_frontier(_md, rfr: float, tickers: tuple, n: int = 60) -> object:
    """tickers is included in the cache key so the frontier invalidates when
    the asset selection changes. _md is excluded (unhashable) but tickers
    gives Streamlit a stable, hashable proxy for the market data."""
    return get_efficient_frontier(_md, n_points=n, risk_free_rate=rfr)


@st.cache_data(show_spinner="Running backtest…")
def cached_backtest(wt: tuple, _md, years: int, rfr: float) -> object:
    return run_backtest(dict(wt), _md, lookback_years=years, risk_free_rate=rfr)


# ── Chart builders ────────────────────────────────────────────────────────────

def frontier_chart(frontier, selected_key: str) -> go.Figure:
    df  = frontier_to_dataframe(frontier)
    fig = go.Figure()

    vol_min = df["volatility"].min() * 100
    vol_max = df["volatility"].max() * 100
    ret_min = df["expected_return"].min() * 100
    ret_max = df["expected_return"].max() * 100

    # ── Sharpe iso-lines (where Sharpe = k, line: ret = rfr + k*vol) ──────────
    rfr_pct = frontier.risk_free_rate * 100
    for sharpe_k, dash, opacity in [(0.5, "dot", 0.18), (1.0, "dash", 0.28), (1.5, "dash", 0.22)]:
        _vols = [vol_min * 0.6, vol_max * 1.15]
        _rets = [rfr_pct + sharpe_k * v for v in _vols]
        fig.add_trace(go.Scatter(
            x=_vols, y=_rets,
            mode="lines",
            line=dict(color=f"rgba(0,208,132,{opacity})", width=1, dash=dash),
            showlegend=False,
            hoverinfo="skip",
            name=f"Sharpe={sharpe_k}",
        ))
        # Label at the right end of each iso-line
        _label_x = min(vol_max * 1.0, _vols[1])
        _label_y = rfr_pct + sharpe_k * _label_x
        if ret_min * 0.8 < _label_y < ret_max * 1.3:
            fig.add_annotation(
                x=_label_x, y=_label_y,
                text=f"  Sharpe {sharpe_k:.1f}",
                showarrow=False,
                font=dict(size=9, color=f"rgba(0,208,132,{opacity * 1.6:.2f})"),
                xanchor="left",
            )

    # ── Gradient fill bands under the frontier ────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["volatility"] * 100, y=df["expected_return"] * 100,
        mode="none", fill="tozeroy",
        fillcolor="rgba(0,208,132,0.03)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["volatility"] * 100, y=df["expected_return"] * 100,
        mode="none", fill="tozeroy",
        fillcolor="rgba(0,208,132,0.025)",
        showlegend=False, hoverinfo="skip",
    ))

    # ── Frontier curve ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = df["volatility"] * 100,
        y    = df["expected_return"] * 100,
        mode = "lines",
        name = "Efficient Frontier",
        line = dict(color=ACCENT, width=2.5, shape="spline", smoothing=1.3),
        hovertemplate = (
            "<b>Frontier</b><br>"
            "Volatility: %{x:.1f}%<br>"
            "Expected Return: %{y:.1f}%<br>"
            "Sharpe: %{customdata:.2f}"
            "<extra></extra>"
        ),
        customdata = df["sharpe_ratio"].values,
    ))

    # ── Individual asset scatter (faint diamonds) ─────────────────────────────
    _ann_ret = frontier_to_dataframe(frontier)  # already a df
    # Extract per-asset return/vol from the market data stored in frontier points
    # We approximate by reading first frontier point weights and the df columns
    _ticker_cols = [c for c in df.columns
                    if c not in ("volatility", "expected_return", "sharpe_ratio", "label")]
    if _ticker_cols:
        # Build a map: ticker → (expected_return, volatility) using weighted averages
        # Actually we don't have individual asset vol here, skip per-asset plot
        pass

    # ── Preset portfolio dots ─────────────────────────────────────────────────
    _SYMBOLS = {
        "min_volatility": "diamond",
        "max_sharpe"    : "star",
        "conservative"  : "circle",
        "balanced"      : "circle",
        "aggressive"    : "circle",
    }
    _SIZES = {
        "min_volatility": 13,
        "max_sharpe"    : 15,
        "conservative"  : 12,
        "balanced"      : 12,
        "aggressive"    : 12,
    }

    for key, port in {
        "min_volatility": frontier.min_volatility,
        "max_sharpe"    : frontier.max_sharpe,
        "conservative"  : frontier.conservative,
        "balanced"      : frontier.balanced,
        "aggressive"    : frontier.aggressive,
    }.items():
        selected = (key == selected_key)
        icon, label_text = PRESET_META[key]
        color  = POINT_COLORS[key]
        sym    = _SYMBOLS[key]
        sz     = _SIZES[key]

        # Glow ring for selected point
        if selected:
            fig.add_trace(go.Scatter(
                x=[port.volatility * 100], y=[port.expected_return * 100],
                mode="markers",
                marker=dict(
                    color="rgba(0,0,0,0)",
                    size=sz + 14,
                    symbol=sym,
                    line=dict(color=color, width=1.5),
                    opacity=0.35,
                ),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x    = [port.volatility * 100],
            y    = [port.expected_return * 100],
            mode = "markers+text",
            name = f"{icon} {label_text.split('(')[0].strip()}",
            marker = dict(
                color   = color,
                size    = sz + (6 if selected else 0),
                symbol  = sym,
                line    = dict(color="#0d0d0d" if not selected else "white",
                               width=1.5 if selected else 1),
                opacity = 1.0,
            ),
            text         = [f"  {icon} {port.expected_return*100:.1f}%"] if selected else [""],
            textposition = "middle right",
            textfont     = dict(size=10, color=color, family="JetBrains Mono, monospace"),
            hovertemplate = (
                f"<b>{icon} {label_text}</b><br>"
                f"Expected Return: {port.expected_return*100:.1f}%<br>"
                f"Volatility:      {port.volatility*100:.1f}%<br>"
                f"Sharpe Ratio:    {port.sharpe_ratio:.3f}"
                f"<extra></extra>"
            ),
        ))

    # ── Risk-free rate point ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0], y=[rfr_pct],
        mode="markers",
        name="Risk-Free Rate",
        marker=dict(color="rgba(158,158,158,0.5)", size=7, symbol="cross",
                    line=dict(color="rgba(158,158,158,0.6)", width=1)),
        hovertemplate=f"Risk-Free Rate: {rfr_pct:.2f}%<extra></extra>",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(**_layout(
        height    = 460,
        hovermode = "closest",
        title     = dict(
            text  = (
                "<b style='color:#e0e0e0'>Efficient Frontier</b>"
                "<br><span style='font-size:11px;color:#2a2a2a'>"
                "Each point on the curve is a mathematically optimal portfolio — "
                "no portfolio below the line offers the same return at lower risk"
                "</span>"
            ),
            font  = dict(size=13, color="#555"),
            pad   = dict(b=12),
            x     = 0,
        ),
        xaxis = dict(
            title      = "Annualised Volatility  (%)",
            ticksuffix = "%",
            showgrid   = True,
            gridwidth  = 1,
            zeroline   = False,
            range      = [max(0, vol_min * 0.75), vol_max * 1.18],
        ),
        yaxis = dict(
            title      = "Annualised Expected Return  (%)",
            ticksuffix = "%",
            showgrid   = True,
            gridwidth  = 1,
            zeroline   = False,
            range      = [min(ret_min * 0.75, rfr_pct * 0.5), ret_max * 1.18],
        ),
        legend = dict(
            orientation = "h",
            y           = -0.18,
            x           = 0,
            font        = dict(size=11, color="#555"),
            bgcolor     = "rgba(0,0,0,0)",
        ),
    ))
    return fig



def pie_chart(portfolio, labels: dict) -> go.Figure:
    ws  = weights_to_series(portfolio, labels)
    fig = go.Figure(go.Pie(
        labels           = list(ws.index),
        values           = [v * 100 for v in ws.values],
        hole             = 0.55,
        marker           = dict(
            colors = PIE_COLORS[:len(ws)],
            line   = dict(color="#080c18", width=2),
        ),
        textinfo              = "label+percent",
        textfont              = dict(size=11, color="#c8e6f5"),
        hovertemplate         = "<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        insidetextorientation = "radial",
    ))
    fig.update_layout(**_layout(
        annotations = [dict(
            text       = portfolio.label,
            x=0.5, y=0.5,
            font_size  = 12,
            font_color = "#c8e6f5",
            showarrow  = False,
        )],
        showlegend = False,
        height     = 300,
        margin     = dict(l=10, r=10, t=10, b=10),
        xaxis      = dict(visible=False),
        yaxis      = dict(visible=False),
    ))
    return fig


def cumulative_chart(bt, label: str) -> go.Figure:
    port = bt.portfolio_cumulative * 100
    bm   = bt.benchmark_cumulative * 100
    up   = bt.total_return >= 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bm.index, y=bm.values,
        mode="lines", name="NIFTY 50",
        line=dict(color="rgba(136,153,187,0.5)", width=1.5, dash="dot"),
        hovertemplate="NIFTY 50: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=port.index, y=port.values,
        mode="lines", name=label,
        line=dict(color=SUCCESS if up else DANGER, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'16,222,160' if up else '248,113,113'},0.06)",
        hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.1)", line_width=1)
    fig.update_layout(**_layout(
        xaxis_title = "Date",
        yaxis_title = "Cumulative Return (%)",
        height      = 340,
        hovermode   = "x unified",
    ))
    return fig


def comparison_chart(bt_a, lbl_a: str, bt_b, lbl_b: str) -> go.Figure:
    fig = go.Figure()
    bm  = bt_a.benchmark_cumulative * 100
    fig.add_trace(go.Scatter(x=bm.index, y=bm.values, mode="lines", name="NIFTY 50",
                             line=dict(color="rgba(136,153,187,0.4)", width=1.5, dash="dot")))
    for bt, lbl, color in [(bt_a, lbl_a, ACCENT), (bt_b, lbl_b, GOLD)]:
        p = bt.portfolio_cumulative * 100
        fig.add_trace(go.Scatter(x=p.index, y=p.values, mode="lines", name=lbl,
                                 line=dict(color=color, width=2.5)))
    fig.update_layout(**_layout(
        xaxis_title="Date", yaxis_title="Cumulative Return (%)",
        height=320, hovermode="x unified",
    ))
    return fig


# ── Metric card HTML ──────────────────────────────────────────────────────────

def mcard(label: str, value: str, tone: str = "") -> str:
    """tone: 'pos' | 'neg' | 'info' | ''"""
    return (
        f'<div class="mcard">'
        f'<div class="lbl">{label}</div>'
        f'<div class="val {tone}">{value}</div>'
        f'</div>'
    )


def sec(icon: str, title: str) -> str:
    return f'<div class="sec-header">{icon}&nbsp;{title}</div>'


# ── Session state (must be before sidebar widget reads) ──────────────────────

if "custom_tickers" not in st.session_state:
    st.session_state.custom_tickers = []
if "sel_key" not in st.session_state:
    st.session_state.sel_key = "balanced"
if "custom_idx" not in st.session_state:
    st.session_state.custom_idx = 0
if "current_holdings" not in st.session_state:
    st.session_state.current_holdings = {}
if "holdings_ticker_set" not in st.session_state:
    st.session_state.holdings_ticker_set = ()

# ── Restore saved portfolio on first load of this browser session ─────────────
# "portfolio_loaded" acts as a one-shot flag: we query the DB exactly once
# per session (not on every Streamlit rerun) to avoid unnecessary disk reads.
if "portfolio_loaded" not in st.session_state:
    _saved = _db.load_portfolio(_USER_ID)
    if _saved:
        # Restore confirmed tickers + params
        st.session_state.confirmed_tickers = tuple(_saved["tickers"])
        st.session_state.confirmed_params  = (_saved["period_years"], _saved["rfr"])
        st.session_state.sel_key           = _saved.get("preset", "balanced")
        st.session_state.current_holdings  = _saved["holdings"]
        # Seed the multiselect so the sidebar shows the restored selection
        st.session_state.asset_multiselect = list(_saved["tickers"])
        st.session_state._portfolio_saved_at = _saved["updated_at"]
    else:
        # New visitor — use defaults
        st.session_state.confirmed_tickers = tuple(DEFAULT_SELECTION)
        st.session_state.confirmed_params  = (5, 0.065)
        st.session_state._portfolio_saved_at = None
    st.session_state.portfolio_loaded = True



# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0; border-bottom:1px solid rgba(0,212,255,0.08); margin-bottom:1rem;">
      <div style="font-size:1.1rem;font-weight:800;color:#e8f4fd;letter-spacing:-0.02em;">■ INVESTIGATE</div>
      <div style="font-size:0.7rem;color:#445577;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.2rem;">Frontier Portfolio Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.15em;color:#00d4ff;font-weight:600;margin-bottom:0.5rem;">Configuration</div>', unsafe_allow_html=True)

    # Asset universe (built-in + custom)
    _custom  = st.session_state.custom_tickers
    _pool    = {**ALL_ASSETS, **{t: t for t in _custom if t not in ALL_ASSETS}}
    _default = [t for t in DEFAULT_SELECTION + _custom if t in _pool]

    # Seed session state for the multiselect on first run only
    if "asset_multiselect" not in st.session_state:
        st.session_state.asset_multiselect = _default

    # ── Unified Search / Add callbacks ─────────────────────────────────────────
    def _handle_master_search():
        _raw = st.session_state.get("master_asset_search", "").strip()
        if not _raw: return
        _upper = _raw.upper()
        
        # Determine the ticker based on user input
        _t = None
        if _upper in _pool:
            _t = _upper
        else:
            # Fuzzy name match first
            _matches = [k for k, v in _pool.items() if _raw.lower() in v.lower() or _raw.lower() in k.lower()]
            if _matches:
                _t = _matches[0]
            else:
                # Treat as new custom ticker
                _t = _upper if _upper.endswith(".NS") else _upper + ".NS"
                if _t not in st.session_state.custom_tickers and _t not in ALL_ASSETS:
                    import yfinance as yf
                    try:
                        if yf.Ticker(_t).history(period="1d").empty:
                            st.toast(f"Asset '{_raw}' not found on NSE.")
                            return
                    except Exception:
                        st.toast(f"Error validating '{_raw}'. It may be invalid.")
                        return
                    st.session_state.custom_tickers.append(_t)

        if _t:
            _cur = list(st.session_state.get("asset_multiselect", []))
            if _t not in _cur:
                _cur.append(_t)
            st.session_state.asset_multiselect = _cur
            st.session_state.master_asset_search = ""

    # Rebuild _pool locally to include any freshly added custom_tickers before rendering multiselect
    _pool = {**ALL_ASSETS, **{t: t for t in st.session_state.custom_tickers if t not in ALL_ASSETS}}

    # ── Unified Search Bar ────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;'
        'color:#445577;font-weight:600;margin:0.2rem 0 0.4rem 0;">'
        '+ Search or Add Asset</div>',
        unsafe_allow_html=True,
    )
    _ca, _cb = st.columns([4, 1])
    with _ca:
        st.text_input(
            "Search or Add Asset",
            placeholder      = "Type name or ticker...",
            label_visibility = "collapsed",
            key              = "master_asset_search",
            on_change        = _handle_master_search,
        )
    with _cb:
        st.button("Add", key="add_master_btn", use_container_width=True, on_click=_handle_master_search)

    # ── Selected Assets Multiselect ───────────────────────────────────────────
    _la, _lb = st.columns([6, 1])
    with _la:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#9e9e9e;margin-bottom:2px;">'
            f'Selected Assets&nbsp;<span style="color:#333;">({len(_pool)} available)</span></div>',
            unsafe_allow_html=True,
        )
    with _lb:
        if st.button("✕", key="clear_assets", use_container_width=True, help="Clear all selections"):
            st.session_state.asset_multiselect = []
            st.rerun()

    sel_tickers = st.multiselect(
        "Selected Assets",
        options          = list(_pool.keys()),
        key              = "asset_multiselect",
        format_func      = lambda t: _pool.get(t, t),
        label_visibility = "collapsed",
    )

    if len(sel_tickers) < 2:
        st.error("Select at least 2 assets to compute a frontier.")
        st.stop()
    if len(sel_tickers) > 30:
        st.warning(f"{len(sel_tickers)} assets selected — optimization may take a few seconds.")

    period_years = st.select_slider("Historical Period (years)", options=[3, 5, 7, 10], value=5)
    rfr_pct      = st.slider("Risk-Free Rate (%)", 4.0, 9.0, 6.5, 0.25,
                              help="India 91-day T-bill yield used in Sharpe computation.")
    rfr          = rfr_pct / 100
    bt_years     = st.select_slider("Backtest Window (years)", options=[1, 2, 3, 5], value=5)

    # ── Analyze button (commits selection; graphs only update on click) ─────────
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    _selection_changed = tuple(sorted(sel_tickers)) != tuple(sorted(st.session_state.confirmed_tickers)) \
                         or (period_years, rfr) != st.session_state.confirmed_params

    _btn_label = "▶\u2009Analyze Portfolio" if not _selection_changed else "▶\u2009Analyze Portfolio  ●"
    _btn_help  = (
        "Run optimization and backtest for the current asset selection."
        if not _selection_changed
        else "Your selection has changed — click to update all charts."
    )
    if st.button(_btn_label, key="analyze_btn", use_container_width=True, help=_btn_help):
        st.session_state.confirmed_tickers = tuple(sel_tickers)
        st.session_state.confirmed_params  = (period_years, rfr)
        st.session_state.sel_key           = "balanced"   # reset preset on new analysis
        st.rerun()

    if _selection_changed:
        st.markdown(
            '<div style="font-size:0.7rem;color:#e53935;margin-top:0.25rem;text-align:center;">'
            '[!] Selection changed — click Analyze to update</div>',
            unsafe_allow_html=True,
        )

    force = st.button("Refresh Market Data", use_container_width=True,
                      help="Bypass 24h cache and re-fetch from Yahoo Finance.")

    # Resolve final params from confirmed state
    _cy, _cr = st.session_state.confirmed_params
    period_years = _cy
    rfr          = _cr
    rfr_pct      = rfr * 100

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.15em;color:#00d4ff;font-weight:600;margin-bottom:0.5rem;">Compare Mode</div>', unsafe_allow_html=True)
    compare = st.toggle("Enable side-by-side comparison", value=False)
    if compare:
        cmp_key = st.selectbox(
            "Second portfolio",
            options=list(PRESET_META.keys()),
            format_func=lambda k: PRESET_META[k][0],
            index=2,
        )

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;'
        'color:#00d084;font-weight:600;margin-bottom:0.5rem;">Save Portfolio</div>',
        unsafe_allow_html=True,
    )

    # Last saved indicator
    _saved_at = st.session_state.get("_portfolio_saved_at")
    if _saved_at:
        try:
            from datetime import datetime as _dt
            _ts = _dt.fromisoformat(_saved_at).strftime("%d %b %Y, %H:%M")
            _saved_label = f"Last saved: {_ts}"
        except Exception:
            _saved_label = "Portfolio saved"
        st.markdown(
            f'<div style="font-size:0.68rem;color:#2a6644;margin-bottom:0.4rem;">'
            f'✓ {_saved_label}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.68rem;color:#333;margin-bottom:0.4rem;">'
            'Not saved yet — your portfolio resets on refresh.</div>',
            unsafe_allow_html=True,
        )

    _sc1, _sc2 = st.columns([3, 2])
    with _sc1:
        if st.button("Save Portfolio", key="save_btn", use_container_width=True,
                     help="Save your current asset selection and holdings to your personal URL."):
            _db.save_portfolio(
                user_id      = _USER_ID,
                tickers      = list(st.session_state.confirmed_tickers),
                holdings     = st.session_state.current_holdings,
                period_years = period_years,
                rfr          = rfr,
                preset       = st.session_state.sel_key,
            )
            from datetime import datetime as _dt
            st.session_state._portfolio_saved_at = _dt.now().isoformat()
            st.toast("Portfolio saved! Bookmark this URL to return to your portfolio.")
            st.rerun()
    with _sc2:
        if st.button("Clear", key="clear_save_btn", use_container_width=True,
                     help="Delete your saved portfolio from the server."):
            _db.delete_portfolio(_USER_ID)
            st.session_state._portfolio_saved_at = None
            st.toast("Saved data cleared.")
            st.rerun()

    st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="disc">
      [!] <strong>Educational tool only.</strong> Not financial advice.
      Past performance does not guarantee future results.
      Consult a SEBI-registered advisor before investing.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA + FRONTIER  (uses confirmed_tickers, not the live multiselect)
# ══════════════════════════════════════════════════════════════════════════════

_active_tickers = st.session_state.confirmed_tickers
md       = load_data(_active_tickers, period_years, force)
frontier = compute_frontier(md, rfr, tickers=_active_tickers)
# Use _pool for labels so custom tickers (not in ALL_ASSETS) get their ticker as the label
labels   = {t: _pool.get(t, t) for t in md.tickers}


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding: 2rem 0 1.5rem 0;">
  <div style="font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; color: #ffffff; line-height: 1.1; margin: 0;">INVESTIGATE</div>
  <div style="font-size: 0.75rem; color: #757575; font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.45rem;">Frontier Portfolio Engine &nbsp;·&nbsp; Indian Equities &nbsp;·&nbsp; Modern Portfolio Theory</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="status-bar">
  <span>Data</span> {md.fetched_at.strftime('%d %b %Y, %H:%M')}
  &nbsp;&nbsp;<span>Assets</span> {len(md.tickers)}
  &nbsp;&nbsp;<span>Trading Days</span> {len(md.prices):,}
  &nbsp;&nbsp;<span>Period</span> {period_years}y window
  &nbsp;&nbsp;<span>Risk-Free Rate</span> {rfr_pct:.2f}%
</div>
""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(sec("◈", "Efficient Frontier"), unsafe_allow_html=True)

st.plotly_chart(
    frontier_chart(frontier, st.session_state.sel_key),
    use_container_width=True,
    key="main_frontier",
)

# Preset buttons
bcols = st.columns(5)
for col, key in zip(bcols, PRESET_META.keys()):
    icon, desc = PRESET_META[key]
    with col:
        if st.button(icon, key=f"btn_{key}", help=desc, use_container_width=True):
            st.session_state.sel_key = key
            st.rerun()

# Custom frontier slider
with st.expander("Select any point on the frontier"):
    n  = len(frontier.frontier_points)
    ix = st.slider("Frontier index (left = lower risk, right = higher return)",
                   0, n - 1, n // 2, label_visibility="collapsed")
    if st.button("Use this point", key="use_custom"):
        st.session_state.sel_key = "_custom"
        st.session_state.custom_idx = ix
        st.rerun()

# Resolve selected portfolio
if st.session_state.sel_key == "_custom":
    sel = frontier.frontier_points[st.session_state.get("custom_idx", len(frontier.frontier_points)//2)]
    sel = sel._replace(label="Custom")
else:
    sel = getattr(frontier, st.session_state.sel_key)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PORTFOLIO DETAILS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(sec("◉", f"Portfolio Details — {sel.label}"), unsafe_allow_html=True)

col_pie, col_info = st.columns([1, 1], gap="large")

with col_pie:
    st.plotly_chart(pie_chart(sel, labels), use_container_width=True, key="main_pie")

with col_info:
    exp_ret = sel.expected_return
    vol     = sel.volatility
    sharpe  = sel.sharpe_ratio

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(mcard("Expected Return", f"{exp_ret*100:+.1f}%",
                    "pos" if exp_ret > 0 else "neg"), unsafe_allow_html=True)
    with c2:
        st.markdown(mcard("Volatility", f"{vol*100:.1f}%", "info"), unsafe_allow_html=True)
    with c3:
        st.markdown(mcard("Sharpe", f"{sharpe:.3f}",
                    "pos" if sharpe > 1 else ("neg" if sharpe < 0 else "")), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;color:#445577;font-weight:600;margin-bottom:0.5rem;">Allocation Weights</div>', unsafe_allow_html=True)

    ws = weights_to_series(sel, labels)
    df_w = ws.reset_index()
    df_w.columns = ["Asset", "Weight"]
    df_w["Allocation"] = (df_w["Weight"] * 100).map("{:.1f}%".format)
    df_w = df_w[["Asset", "Allocation"]].sort_values("Allocation", ascending=False)
    st.dataframe(df_w, use_container_width=True, hide_index=True, height=195)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(sec("◐", f"Historical Backtest — {bt_years}y window"), unsafe_allow_html=True)
st.markdown(
    f'<div style="font-size:0.78rem;color:#445577;margin-bottom:0.75rem;">'
    f'Simulating <b style="color:#8899bb">{sel.label}</b> allocation held over the past '
    f'<b style="color:#8899bb">{bt_years} years</b>, rebalanced daily, vs. NIFTY 50.</div>',
    unsafe_allow_html=True,
)

wt      = tuple(sorted(sel.weights.items()))
bt      = cached_backtest(wt, md, bt_years, rfr)

st.plotly_chart(cumulative_chart(bt, sel.label), use_container_width=True, key="main_bt")

# Metrics row
m1, m2, m3, m4, m5 = st.columns(5)
outperforms = bt.total_return > bt.benchmark_total_return
delta = abs(bt.total_return - bt.benchmark_total_return)

for col, lbl, val, tone in [
    (m1, "Total Return",  f"{bt.total_return*100:+.1f}%",      "pos" if bt.total_return > 0 else "neg"),
    (m2, "CAGR",          f"{bt.annualized_return*100:+.1f}%", "pos" if bt.annualized_return > 0 else "neg"),
    (m3, "Sharpe Ratio",  f"{bt.sharpe_ratio:.3f}",            "pos" if bt.sharpe_ratio > 1 else ""),
    (m4, "Max Drawdown",  f"{bt.max_drawdown*100:.1f}%",       "neg"),
    (m5, "vs Benchmark",  f"{'▲' if outperforms else '▼'} {delta*100:.1f}pp",
                                                                "pos" if outperforms else "neg"),
]:
    with col:
        st.markdown(mcard(lbl, val, tone), unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
with st.expander("Full statistics table"):
    st.dataframe(backtest_stats_table(bt), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3b — REBALANCING GUIDE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(sec("⇌", "My Current Portfolio — Rebalancing Guide"), unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:0.8rem;color:#555;margin-bottom:1rem;">'
    'Enter how much you currently have invested in each stock (₹ value). '
    'The app compares your current allocation against the '
    f'<b style="color:#e0e0e0">{sel.label}</b> optimal target and tells you '
    'what to Add, Trim, or Hold — sorted by priority.'
    '</div>',
    unsafe_allow_html=True,
)

# Reset holdings inputs whenever the analyzed stock set changes
_cur_ticker_set = tuple(sorted(md.tickers))
if st.session_state.holdings_ticker_set != _cur_ticker_set:
    st.session_state.current_holdings   = {t: 0 for t in md.tickers}
    st.session_state.holdings_ticker_set = _cur_ticker_set

# ── Input grid ────────────────────────────────────────────────────────────────
_already_filled = sum(st.session_state.current_holdings.get(t, 0) for t in md.tickers) > 0
with st.expander("Enter / update your current holdings (₹)", expanded=not _already_filled):
    st.markdown(
        '<div style="font-size:0.75rem;color:#444;margin-bottom:0.75rem;">'
        'Enter the <b style="color:#9e9e9e">current market value in ₹</b> of each position. '
        "Leave as 0 for stocks you don't currently hold.</div>",
        unsafe_allow_html=True,
    )
    _chunks = [md.tickers[i:i+3] for i in range(0, len(md.tickers), 3)]
    for _chunk in _chunks:
        _icols = st.columns(3)
        for _icol, _ticker in zip(_icols, _chunk):
            with _icol:
                _lbl  = labels.get(_ticker, _ticker)
                _disp = _lbl if len(_lbl) <= 20 else _lbl[:18] + "…"
                _prev = int(st.session_state.current_holdings.get(_ticker, 0))
                _v    = st.number_input(
                    _disp,
                    min_value = 0,
                    value     = _prev,
                    step      = 1000,
                    key       = f"hld_{_ticker}",
                    help      = f"{_ticker}",
                )
                st.session_state.current_holdings[_ticker] = float(_v)

# ── Results ───────────────────────────────────────────────────────────────────
_holdings      = st.session_state.current_holdings
_total_invested = sum(_holdings.get(t, 0) for t in md.tickers)

if _total_invested <= 0:
    st.markdown(
        '<div style="text-align:center;padding:2rem;color:#2a2a2a;font-size:0.85rem;">'
        'Enter your current holdings above to see personalised rebalancing guidance.</div>',
        unsafe_allow_html=True,
    )
else:
    # Weights from user input and from selected optimal portfolio
    _cur_w = {t: _holdings.get(t, 0) / _total_invested for t in md.tickers}
    _tgt_w = {t: sel.weights.get(t, 0.0) for t in md.tickers}

    HOLD_BAND = 4.0   # ±4 pp = "Hold"
    _rows = []
    for _t in md.tickers:
        _cp = _cur_w.get(_t, 0.0) * 100
        _tp = _tgt_w.get(_t, 0.0) * 100
        _d  = _tp - _cp
        _cv = _holdings.get(_t, 0.0)
        if _tp == 0 and _cp == 0:
            continue
        if _d > HOLD_BAND:
            _act, _col, _ico = "Add",  "#00d084", "▲"
        elif _d < -HOLD_BAND:
            _act, _col, _ico = "Trim", "#e53935", "▼"
        else:
            _act, _col, _ico = "Hold", "#9e9e9e", "—"
        _rows.append(dict(
            ticker=_t, name=labels.get(_t, _t),
            cur_pct=_cp, tgt_pct=_tp, delta=_d,
            cur_val=_cv, action=_act, color=_col, icon=_ico,
        ))
    _rows.sort(key=lambda r: abs(r["delta"]), reverse=True)

    # Compute current portfolio Sharpe
    _wt_arr  = np.array([_cur_w.get(t, 0) for t in md.tickers])
    _wt_arr /= _wt_arr.sum() if _wt_arr.sum() > 0 else 1
    _ret_arr = md.annualized_returns.reindex(md.tickers).fillna(0).values
    _cov_mat = md.covariance_matrix.reindex(index=md.tickers, columns=md.tickers).fillna(0).values
    _cur_er  = float(np.dot(_wt_arr, _ret_arr))
    _cur_vol = float(np.sqrt(_wt_arr @ _cov_mat @ _wt_arr))
    _cur_sh  = (_cur_er - rfr) / _cur_vol if _cur_vol > 0 else 0.0
    _tgt_sh  = sel.sharpe_ratio
    _sh_gain = _tgt_sh - _cur_sh

    # Summary metrics
    rb1, rb2, rb3, rb4 = st.columns(4)
    for _col_w, _lbl_w, _val_w, _tone_w in [
        (rb1, "Portfolio Value",  f"₹{_total_invested:,.0f}",                   "info"),
        (rb2, "Your Sharpe",     f"{_cur_sh:.3f}",  "pos" if _cur_sh > 1 else ("neg" if _cur_sh < 0 else "")),
        (rb3, "Target Sharpe",   f"{_tgt_sh:.3f}",  "pos" if _tgt_sh > 1 else ""),
        (rb4, "Sharpe Δ",        f"{'▲' if _sh_gain >= 0 else '▼'} {abs(_sh_gain):.3f}",
                                              "pos" if _sh_gain >= 0 else "neg"),
    ]:
        with _col_w:
            st.markdown(mcard(_lbl_w, _val_w, _tone_w), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Top-3 priority callout
    _pri_html = "".join([
        f'<span style="background:rgba('
        f'{"0,208,132" if r["action"]=="Add" else "229,57,53" if r["action"]=="Trim" else "158,158,158"}'
        f',0.1);border:1px solid rgba('
        f'{"0,208,132" if r["action"]=="Add" else "229,57,53" if r["action"]=="Trim" else "158,158,158"}'
        f',0.3);border-radius:4px;padding:0.25rem 0.65rem;font-size:0.77rem;'
        f'color:{r["color"]};font-weight:600;white-space:nowrap;">'
        f'{r["icon"]} {r["action"]} {r["name"]} ({r["delta"]:+.1f}pp)</span>'
        for r in _rows[:3]
    ])
    st.markdown(
        '<div style="font-size:0.68rem;color:#444;text-transform:uppercase;'
        'letter-spacing:0.1em;font-weight:600;margin-bottom:0.4rem;">Top Priority Actions</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1.25rem;">{_pri_html}</div>',
        unsafe_allow_html=True,
    )

    # Horizontal bar chart: current vs target
    _names = [r["name"] if len(r["name"]) <= 22 else r["name"][:20]+"…" for r in _rows]
    fig_rb = go.Figure()
    fig_rb.add_trace(go.Bar(
        name="Current", y=_names, x=[r["cur_pct"] for r in _rows],
        orientation="h",
        marker_color="rgba(255,255,255,0.1)",
        marker_line=dict(color="rgba(255,255,255,0.18)", width=1),
        hovertemplate="%{y}: %{x:.1f}%<extra>Current</extra>",
    ))
    fig_rb.add_trace(go.Bar(
        name=f"Target ({sel.label})", y=_names, x=[r["tgt_pct"] for r in _rows],
        orientation="h",
        marker_color=[r["color"] for r in _rows], opacity=0.65,
        hovertemplate="%{y}: %{x:.1f}%<extra>Target</extra>",
    ))
    fig_rb.update_layout(_layout(
        barmode="overlay",
        height=max(260, len(_rows) * 34 + 80),
        title=dict(text="Current vs Target Allocation", font=dict(size=13, color="#555")),
        xaxis=dict(title="Weight (%)", ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=10, r=20, t=60, b=35),
    ))
    st.plotly_chart(fig_rb, use_container_width=True, key="rebalance_chart")

    # Full action table
    st.markdown(
        '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'color:#444;font-weight:600;margin:0.75rem 0 0.35rem 0;">Full Rebalancing Table</div>',
        unsafe_allow_html=True,
    )
    _tbl = pd.DataFrame([{
        "Stock":          r["name"],
        "You Hold":       f"{r['cur_pct']:.1f}%",
        "Current Value":  f"₹{r['cur_val']:,.0f}" if r["cur_val"] > 0 else "—",
        "Target":         f"{r['tgt_pct']:.1f}%",
        "Δ (pp)":         f"{r['delta']:+.1f}pp",
        "₹ to Buy/Sell":  (
            f"+₹{abs(r['delta'] / 100 * _total_invested):,.0f}"
            if r["action"] == "Add"
            else f"-₹{abs(r['delta'] / 100 * _total_invested):,.0f}"
            if r["action"] == "Trim"
            else "Hold"
        ),
        "Action":         f"{r['icon']} {r['action']}",
    } for r in _rows])
    st.dataframe(
        _tbl,
        use_container_width=True,
        hide_index=True,
        height=min(420, 44 + len(_rows) * 36),
    )
    st.markdown(
        '<div style="font-size:0.67rem;color:#252525;margin-top:0.35rem;">'
        'pp = percentage points. '
        '₹ to Buy/Sell = Δpp ÷ 100 × total portfolio value. '
        'Add / Trim threshold: ±4pp. '
        'For educational purposes only — not financial advice.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMPARISON (optional)
# ══════════════════════════════════════════════════════════════════════════════

if compare:
    st.markdown(sec("⇄", f"Comparison — {sel.label}  vs  {PRESET_META[cmp_key][0]}"), unsafe_allow_html=True)

    port_b = getattr(frontier, cmp_key)
    wt_b   = tuple(sorted(port_b.weights.items()))
    bt_b   = cached_backtest(wt_b, md, bt_years, rfr)

    st.plotly_chart(comparison_chart(bt, sel.label, bt_b, port_b.label), use_container_width=True, key="cmp_cumulative")

    ca, cb = st.columns(2, gap="large")

    def render_col(col, portfolio, result, col_id: str):
        with col:
            st.markdown(f'<div style="font-size:0.8rem;font-weight:600;color:#c8e6f5;margin-bottom:0.5rem;">{portfolio.label}</div>', unsafe_allow_html=True)
            st.plotly_chart(pie_chart(portfolio, labels), use_container_width=True, key=f"cmp_pie_{col_id}")
            g1, g2 = st.columns(2)
            metrics = [
                ("Total Return", f"{result.total_return*100:+.1f}%",      "pos" if result.total_return > 0 else "neg"),
                ("CAGR",         f"{result.annualized_return*100:+.1f}%", "pos" if result.annualized_return > 0 else "neg"),
                ("Sharpe",       f"{result.sharpe_ratio:.3f}",            "pos" if result.sharpe_ratio > 1 else ""),
                ("Max DD",       f"{result.max_drawdown*100:.1f}%",       "neg"),
            ]
            for i, (lbl, val, tone) in enumerate(metrics):
                with (g1 if i % 2 == 0 else g2):
                    st.markdown(mcard(lbl, val, tone), unsafe_allow_html=True)
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    render_col(ca, sel,    bt, "a")
    render_col(cb, port_b, bt_b, "b")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HOW IT WORKS + CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(sec("◌", "How to Use Investigate"), unsafe_allow_html=True)

st.markdown("""
<div class="how-grid">

  <div class="how-card">
    <div class="step-num">01</div>
    <div class="step-title">Build Your Universe</div>
    <div class="step-body">
      Select from the full <b>Nifty 50</b> universe plus gold &amp; debt ETFs in the sidebar.
      Mix sectors — financials, IT, pharma, consumer — for genuine diversification.
      Start with the default 9-asset set, then expand.
    </div>
  </div>

  <div class="how-card">
    <div class="step-num">02</div>
    <div class="step-title">Read the Frontier</div>
    <div class="step-body">
      The curve shows every <b>mathematically optimal portfolio</b> for your chosen assets.
      Nothing above the curve is achievable — it's the physical limit of diversification.
      Each dot on the curve is a different allocation of your assets.
    </div>
  </div>

  <div class="how-card">
    <div class="step-num">03</div>
    <div class="step-title">Choose Your Risk Level</div>
    <div class="step-body">
      Pick <b>Conservative</b> for capital safety, <b>Balanced</b> for growth,
      or <b>Aggressive</b> for maximum potential return. Or use the frontier slider
      to land anywhere between — each position has an exact allocation breakdown.
    </div>
  </div>

  <div class="how-card">
    <div class="step-num">04</div>
    <div class="step-title">Validate with History</div>
    <div class="step-body">
      The backtest shows how your allocation <b>actually would have performed</b>
      over the past 1–5 years, including crashes and recoveries — compared to
      the NIFTY 50 benchmark. Use it to gut-check your risk tolerance.
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.15em;color:#818cf8;font-weight:600;margin-bottom:0.5rem;">Key Concepts</div>', unsafe_allow_html=True)

st.markdown("""
<div class="concept-grid">

  <div class="concept-card">
    <div class="c-title">Sharpe Ratio</div>
    <div class="c-body">
      Measures return earned <i>per unit of risk taken</i>, above the risk-free rate.
      <b>&gt; 1.0</b> is good; <b>&gt; 2.0</b> is excellent.
      The Max Sharpe portfolio is mathematically the most efficient allocation.
    </div>
  </div>

  <div class="concept-card">
    <div class="c-title">Maximum Drawdown</div>
    <div class="c-body">
      The worst peak-to-trough loss in the period — your real pain at the worst moment.
      A <b>−20% drawdown</b> means your portfolio fell 20% from its high before recovering.
      Conservative portfolios have smaller drawdowns.
    </div>
  </div>

  <div class="concept-card">
    <div class="c-title">CAGR</div>
    <div class="c-body">
      Compound Annual Growth Rate — the equivalent constant yearly return that gives
      you the same total gain. More meaningful than total return alone because it
      accounts for how long the investment was held.
    </div>
  </div>

  <div class="concept-card">
    <div class="c-title">Efficient Frontier</div>
    <div class="c-body">
      The set of portfolios that maximize return for every level of risk, discovered by
      Harry Markowitz in 1952. Portfolios <i>below</i> the frontier are suboptimal —
      you could get more return for the same risk by rebalancing.
    </div>
  </div>

  <div class="concept-card">
    <div class="c-title">Covariance & Correlation</div>
    <div class="c-body">
      Diversification works because assets don't always move together. When gold
      rises, equities may fall. The optimizer uses the covariance matrix to find
      combinations where individual risks partially <i>cancel each other out</i>.
    </div>
  </div>

  <div class="concept-card">
    <div class="c-title">Calmar Ratio</div>
    <div class="c-body">
      CAGR divided by max drawdown. Rewards high return <i>and</i> punishes deep
      losses. A Calmar of <b>0.5+</b> is generally acceptable; <b>1.0+</b> is strong.
      Use it to compare portfolios with very different volatility profiles.
    </div>
  </div>

</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
  INVESTIGATE &nbsp;·&nbsp; Frontier Portfolio Engine &nbsp;·&nbsp;
  Built with Streamlit, PyPortfolioOpt &amp; yfinance &nbsp;·&nbsp;
  Data via Yahoo Finance / NSE
  <br>
  <span style="font-size:0.68rem;">
    ⚠ For educational purposes only. Not financial advice.
    Past performance does not guarantee future results.
  </span>
</div>
""", unsafe_allow_html=True)
