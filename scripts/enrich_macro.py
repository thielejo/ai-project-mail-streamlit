from __future__ import annotations

from pathlib import Path

import pandas as pd


MICRO_PATH = Path("car_prices_clean.csv")
MACRO_PATH = Path("macro_index.csv")
OUTPUT_PATH = Path("car_prices_macro.csv")

# FRED series downloaded directly — no API key required
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

FRED_SERIES = {
    "cpi_used_cars": "CUSR0000SETA01",  # CPI Used Cars & Trucks (MUVVI proxy)
    "fedfunds": "FEDFUNDS",              # US Federal Funds Rate
}

# 2015 is our neutral economic baseline (as defined in the project proposal)
CPI_BASE_YEAR = 2015


def fetch_fred_series(column_name: str, series_id: str) -> pd.Series:
    """Download a single FRED series and return it indexed by year-month period."""
    url = FRED_BASE + series_id
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df["year_month"] = df["observation_date"].dt.to_period("M")
    return df.set_index("year_month")[series_id].rename(column_name)


def build_macro_index() -> pd.DataFrame:
    """
    Fetch all FRED series and compute the CPI multiplier relative to the 2015 baseline.

    The cpi_multiplier is the core of Stage 2:
        Live Price = Stage1_Baseline × cpi_multiplier × seasonal_factor
    """
    series = [fetch_fred_series(name, sid) for name, sid in FRED_SERIES.items()]
    macro = pd.concat(series, axis=1).dropna(how="all")

    baseline_cpi = macro.loc["2015", "cpi_used_cars"].mean()
    macro["cpi_multiplier"] = macro["cpi_used_cars"] / baseline_cpi

    macro.index = macro.index.astype(str)
    return macro.reset_index().rename(columns={"year_month": "year_month"})


def enrich_car_prices(
    micro_path: Path = MICRO_PATH,
    macro_path: Path = MACRO_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    print("1. Lade bereinigten Fahrzeugdatensatz...")
    micro = pd.read_csv(micro_path)
    micro["saledate"] = pd.to_datetime(micro["saledate"], utc=True)
    micro["year_month"] = micro["saledate"].dt.strftime("%Y-%m")
    print(f"   -> {len(micro):,} Fahrzeuge geladen.")

    print("2. Lade Makrodaten von FRED (Internet erforderlich)...")
    macro = build_macro_index()
    macro.to_csv(macro_path, index=False)
    print(f"   -> Makro-Index gespeichert: {macro_path}")
    print(f"   -> Zeitraum: {macro['year_month'].min()} bis {macro['year_month'].max()}")
    print(f"   -> CPI-Basiswert 2015: {macro.loc[macro['year_month'].str.startswith('2015'), 'cpi_used_cars'].mean():.2f}")

    print("3. Verknüpfe Fahrzeug- und Makrodaten über year_month...")
    merged = micro.merge(macro, on="year_month", how="left")
    missing = merged["cpi_used_cars"].isna().sum()
    if missing > 0:
        print(f"   -> Warnung: {missing:,} Zeilen ohne Makrowert (Datumsbereich prüfen).")
    else:
        print(f"   -> Alle {len(merged):,} Zeilen erfolgreich verknüpft.")

    merged.to_csv(output_path, index=False)
    print(f"   -> Ausgabe gespeichert: {output_path}")

    return merged


if __name__ == "__main__":
    enrich_car_prices()
