"""
db.py — Persistence Layer (Option 2: Anonymous UUID + SQLite)
==============================================================
Saves and loads user portfolios keyed by an anonymous UUID
stored in the URL query parameter (?uid=...).

No authentication required. The UUID is the user's identity —
they keep their portfolio by bookmarking their personalised URL.

Migration path to Option 3 (auth):
    Replace user_id (UUID string) with an authenticated email.
    All function signatures stay identical — only the caller changes.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path("portfolios.db")


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_PORTFOLIOS = """
CREATE TABLE IF NOT EXISTS portfolios (
    user_id      TEXT PRIMARY KEY,
    tickers      TEXT    NOT NULL,   -- JSON array  e.g. ["HDFCBANK.NS", ...]
    holdings     TEXT    NOT NULL,   -- JSON object e.g. {"HDFCBANK.NS": 50000, ...}
    period_years INTEGER NOT NULL DEFAULT 5,
    rfr          REAL    NOT NULL DEFAULT 0.065,
    preset       TEXT    NOT NULL DEFAULT 'balanced',
    updated_at   TEXT    NOT NULL
)
"""


# ── Public API ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create database and tables if they don't already exist. Call once at startup."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_PORTFOLIOS)
        conn.commit()
    log.info("Portfolio DB ready at %s", DB_PATH.resolve())


def save_portfolio(
    user_id: str,
    tickers: list[str],
    holdings: dict[str, float],
    period_years: int  = 5,
    rfr: float         = 0.065,
    preset: str        = "balanced",
) -> None:
    """
    Insert or replace a user's portfolio.

    Parameters
    ----------
    user_id      : UUID string from the URL param (or email in Option 3).
    tickers      : Ordered list of NSE tickers in the confirmed selection.
    holdings     : {ticker: rupee_value} mapping of current holdings.
    period_years : Historical lookback window saved alongside the portfolio.
    rfr          : Risk-free rate used in the last analysis.
    preset       : Active preset key (e.g. 'balanced', 'max_sharpe').
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO portfolios
                (user_id, tickers, holdings, period_years, rfr, preset, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                tickers      = excluded.tickers,
                holdings     = excluded.holdings,
                period_years = excluded.period_years,
                rfr          = excluded.rfr,
                preset       = excluded.preset,
                updated_at   = excluded.updated_at
            """,
            (
                user_id,
                json.dumps(tickers),
                json.dumps(holdings),
                period_years,
                rfr,
                preset,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    log.info("Portfolio saved  user=...%s", user_id[-8:])


def load_portfolio(user_id: str) -> dict | None:
    """
    Load a saved portfolio. Returns None if no record exists for this user_id.

    Return schema
    -------------
    {
        "tickers":      list[str],
        "holdings":     dict[str, float],
        "period_years": int,
        "rfr":          float,
        "preset":       str,
        "updated_at":   str,   # ISO-8601 timestamp
    }
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT tickers, holdings, period_years, rfr, preset, updated_at
            FROM portfolios
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "tickers":      json.loads(row[0]),
        "holdings":     {k: float(v) for k, v in json.loads(row[1]).items()},
        "period_years": int(row[2]),
        "rfr":          float(row[3]),
        "preset":       row[4],
        "updated_at":   row[5],
    }


def delete_portfolio(user_id: str) -> None:
    """
    Remove a user's saved portfolio.
    Called when the user clicks "Clear saved data" in the sidebar.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM portfolios WHERE user_id = ?", (user_id,))
        conn.commit()
    log.info("Portfolio deleted  user=...%s", user_id[-8:])
