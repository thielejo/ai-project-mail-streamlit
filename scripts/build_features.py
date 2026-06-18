from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("car_prices_clean.csv")
OUTPUT_PATH = Path("car_prices_features.csv")

BASE_COLUMNS = [
    "year",
    "make",
    "model",
    "body",
    "condition",
    "odometer",
    "saledate",
    "sellingprice",
]


def build_car_price_features(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, usecols=BASE_COLUMNS)
    rows_loaded = len(df)

    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    df = df.dropna(
        subset=[
            "year",
            "make",
            "model",
            "body",
            "condition",
            "odometer",
            "saledate",
            "sellingprice",
        ]
    ).copy()

    df["sale_year"] = df["saledate"].dt.year
    df["sale_month"] = df["saledate"].dt.month
    df["year_month"] = df["saledate"].dt.strftime("%Y-%m")

    df["vehicle_age"] = df["sale_year"] - df["year"]
    df["vehicle_age"] = df["vehicle_age"].clip(lower=0)

    feature_columns = [
        "vehicle_age",
        "year_month",
        "sale_month",
        "make",
        "model",
        "odometer",
        "condition",
        "body",
        "sellingprice",
    ]
    features = df[feature_columns].copy()

    features["vehicle_age"] = features["vehicle_age"].astype("Int64")
    features["sale_month"] = features["sale_month"].astype("Int64")
    features["odometer"] = features["odometer"].astype("Int64")
    features["sellingprice"] = features["sellingprice"].astype("Int64")

    features.to_csv(output_path, index=False)

    print(f"Rows loaded: {rows_loaded:,}")
    print(f"Rows written: {len(features):,}")
    print(f"Rows removed for missing feature values: {rows_loaded - len(features):,}")
    print(f"Output: {output_path}")
    print(f"Columns: {', '.join(features.columns)}")

    return features


if __name__ == "__main__":
    build_car_price_features()
