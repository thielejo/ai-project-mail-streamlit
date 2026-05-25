from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("car_prices.csv")
OUTPUT_PATH = Path("car_prices_clean.csv")

CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "trim",
    "body",
    "transmission",
    "vin",
    "state",
    "color",
    "interior",
    "seller",
]


def read_car_prices(path: Path) -> pd.DataFrame:
    """Read the source CSV and repair rows with one unquoted comma in trim."""
    rows: list[list[str]] = []

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        expected_fields = len(header)

        for line_number, row in enumerate(reader, start=2):
            if len(row) == expected_fields:
                rows.append(row)
                continue

            if len(row) == expected_fields + 1:
                row = row[:3] + [f"{row[3]}, {row[4].strip()}"] + row[5:]
                rows.append(row)
                continue

            raise ValueError(
                f"Unexpected field count in line {line_number}: "
                f"expected {expected_fields}, got {len(row)}"
            )

    return pd.DataFrame(rows, columns=header)


def normalize_category(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    normalized = normalized.str.replace(r"\s+", " ", regex=True)
    normalized = normalized.str.replace("navitgation", "navigation", regex=False)
    return normalized.replace("", pd.NA)


def clean_car_prices(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    df = read_car_prices(input_path)

    numeric_columns = ["year", "condition", "odometer", "mmr", "sellingprice"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before_missing_filter = len(df)
    df = df.dropna(subset=["sellingprice", "odometer"]).copy()

    for column in CATEGORICAL_COLUMNS:
        df[column] = normalize_category(df[column])

    df["transmission"] = df["transmission"].fillna("unknown")

    saledate = df["saledate"].astype("string").str.replace(r" \([A-Z]{3}\)$", "", regex=True)
    df["saledate"] = pd.to_datetime(
        saledate,
        format="%a %b %d %Y %H:%M:%S GMT%z",
        errors="coerce",
        utc=True,
    )
    df["saledate"] = df["saledate"].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    integer_columns = ["year", "odometer", "mmr", "sellingprice"]
    for column in integer_columns:
        df[column] = df[column].astype("Int64")

    df.to_csv(output_path, index=False)

    print(f"Rows loaded: {before_missing_filter:,}")
    print(f"Rows written: {len(df):,}")
    print(f"Rows removed for missing sellingprice/odometer: {before_missing_filter - len(df):,}")
    print(f"Output: {output_path}")

    return df


if __name__ == "__main__":
    clean_car_prices()
