"""Stage 2: CPI-based macro adjustment applied to Stage 1 baseline prices.

Core formula:
    stage2_price = stage1_price × cpi_multiplier(target_month)

The cpi_multiplier is normalised to the 2015 annual average (= 1.0 in the
macro_index.csv produced by enrich_macro.py). Stage 1 was trained on 2014–2015
auction data, so its output represents a 2015-equivalent price level. Multiplying
by the current CPI ratio adjusts that baseline to today's used-car market.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACRO_PATH = PROJECT_ROOT / "macro_index.csv"

MACRO_SIGNAL_LABELS: dict[str, str] = {
    "cpi_used_cars": "CPI Gebrauchtwagen (FRED)",
    "fedfunds": "Leitzins Fed Funds %",
    "consumer_sentiment": "Konsumentenstimmung (Univ. Michigan)",
    "unemployment": "Arbeitslosenquote %",
    "total_vehicle_sales": "Fahrzeugverkäufe Mio. SAAR",
    "recession": "Rezession NBER (0/1)",
    "credit_spread": "High-Yield-Spread %",
}


def load_macro_index(path: Path = MACRO_PATH) -> pd.DataFrame:
    """Load macro_index.csv and return it indexed by year_month strings."""
    df = pd.read_csv(path)
    df["year_month"] = df["year_month"].astype(str)
    return df.set_index("year_month")


def _resolve_year_month(year_month: str, macro: pd.DataFrame) -> str:
    """Return nearest available past month.

    Forward-fills missing months (FRED publication lag, future months).
    Falls back to the earliest available month if year_month precedes the index.
    """
    if year_month in macro.index:
        return year_month
    available = sorted(macro.index.tolist())
    past = [m for m in available if m <= year_month]
    return past[-1] if past else available[0]


def get_cpi_multiplier(year_month: str, macro: pd.DataFrame) -> float:
    resolved = _resolve_year_month(year_month, macro)
    return float(macro.loc[resolved, "cpi_multiplier"])


def get_macro_context(year_month: str, macro: pd.DataFrame) -> dict:
    """Return the full macro snapshot for a given month as a flat dict."""
    resolved = _resolve_year_month(year_month, macro)
    row = macro.loc[resolved]
    context: dict = {
        "year_month": resolved,
        "cpi_multiplier": round(float(row["cpi_multiplier"]), 4),
    }
    for col in MACRO_SIGNAL_LABELS:
        val = row.get(col, np.nan)
        context[col] = round(float(val), 4) if pd.notna(val) else None
    return context


def apply_stage2(
    stage1_price: float,
    year_month: str,
    macro: pd.DataFrame,
) -> tuple[float, float]:
    """Apply CPI adjustment to a Stage 1 baseline price.

    Returns (stage2_price, cpi_multiplier).
    """
    multiplier = get_cpi_multiplier(year_month, macro)
    return stage1_price * multiplier, multiplier
