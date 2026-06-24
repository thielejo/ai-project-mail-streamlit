from __future__ import annotations

from pathlib import Path

import pandas as pd


MICRO_PATH = Path("car_prices_clean.csv")
MACRO_PATH = Path("macro_index.csv")
OUTPUT_PATH = Path("car_prices_macro.csv")

# FRED series downloaded directly — no API key required
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# fredgraph.csv ignores query params like observation_start, so we trim post-download.
# 1996-01 is chosen because: USREC otherwise drags the index back to 1854, and
# credit_spread (BAMLH0A0HYM2) starts in late 1996 — keeping earlier rows
# would leave nearly every macro column NaN before that date.
MACRO_START = pd.Period("1996-01", "M")

FRED_SERIES = {
    # --- Price indices ---
    "cpi_used_cars": "CUSR0000SETA01",   # CPI Used Cars & Trucks (MUVVI proxy)

    # --- Monetary policy ---
    "fedfunds": "FEDFUNDS",              # US Federal Funds Rate

    # --- Consumer demand ---
    "consumer_sentiment": "UMCSENT",    # Univ. of Michigan Consumer Sentiment
    "unemployment": "UNRATE",           # US Unemployment Rate

    # --- Automotive market volume ---
    "total_vehicle_sales": "TOTALSA",   # Total Vehicle Sales (millions, SAAR)

    # --- Energy / external shocks ---
    "oil_price_wti": "DCOILWTICO",      # WTI Crude Oil Price (USD/barrel)

    # --- Crisis / stress indicators ---
    "recession": "USREC",               # NBER Recession Indicator (0 = no, 1 = yes)
    "credit_spread": "BAMLH0A0HYM2",   # High-Yield Credit Spread (market stress proxy)
}

# 2015 is our neutral economic baseline (as defined in the project proposal)
CPI_BASE_YEAR = 2015


def fetch_fred_series(column_name: str, series_id: str, retries: int = 2) -> pd.Series | None:
    """Download a single FRED series. Returns None and warns if all attempts fail."""
    url = FRED_BASE + series_id
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_csv(url, parse_dates=["observation_date"], na_values=["."])
            df["year_month"] = df["observation_date"].dt.to_period("M")
            # Aggregate duplicates (e.g. data revisions) by taking the last value per month
            series = df.groupby("year_month")[series_id].last().rename(column_name)
            return series
        except Exception as e:
            if attempt < retries:
                print(f"   -> Versuch {attempt} fehlgeschlagen für '{column_name}', wiederhole...")
            else:
                print(f"   -> Warnung: '{column_name}' ({series_id}) nicht verfügbar: {e}")
                return None


def build_macro_index() -> pd.DataFrame:
    """
    Fetch all FRED series and compute the CPI multiplier relative to the 2015 baseline.

    Core formula for Stage 2:
        Live Price = Stage1_Baseline × cpi_multiplier × seasonal_factor

    Additional series provide context for the LLM orchestrator and future model iterations.
    """
    series = []
    for name, sid in FRED_SERIES.items():
        result = fetch_fred_series(name, sid)
        if result is not None:
            series.append(result)
            print(f"   -> '{name}' geladen ({sid})")

    macro = pd.concat(series, axis=1).sort_index().dropna(how="all")

    # Trim to MACRO_START: fredgraph.csv ignores observation_start params, so USREC
    # would otherwise drag the index back to 1854.
    macro = macro[macro.index >= MACRO_START]

    # Forward-fill within-series gaps (quarterly surveys with monthly index,
    # occasional missing months in continuous series).
    macro = macro.ffill()

    baseline_cpi = macro.loc["2015", "cpi_used_cars"].mean()
    macro["cpi_multiplier"] = macro["cpi_used_cars"] / baseline_cpi

    macro.index = macro.index.astype(str)
    return macro.reset_index().rename(columns={"index": "year_month"})


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
    print(f"   -> Spalten: {list(macro.columns)}")
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
