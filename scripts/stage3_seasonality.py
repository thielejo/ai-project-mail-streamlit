"""Stage 3: conservative seasonal adjustment by body type and sale month.

The rule compares CPI-normalized auction prices with Stage-1 predictions made
for one fixed reference period. This controls for vehicle mix and prevents the
target month from being counted once in Stage 1 and again in Stage 3.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "car_prices_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "price_model.joblib"
MACRO_PATH = PROJECT_ROOT / "macro_index.csv"
SEASONALITY_PATH = PROJECT_ROOT / "models" / "seasonality_factors.csv"

FEATURE_COLUMNS = [
    "vehicle_age",
    "sale_month",
    "odometer",
    "condition",
    "year_month",
    "make",
    "model",
    "body",
]
REFERENCE_YEAR_MONTH = "2015-02"
REFERENCE_MONTH = 2

MONTH_NAMES: dict[int, str] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

FACTOR_MIN = 0.85
FACTOR_MAX = 1.15
SHRINKAGE_OBSERVATIONS = 1_000
MIN_RECOMMENDATION_OBSERVATIONS = 100
MIN_RECOMMENDATION_MONTHS = 2


def prepare_seasonality_data(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    macro_path: Path = MACRO_PATH,
) -> pd.DataFrame:
    """Create vehicle-mix and CPI-adjusted rows used to estimate seasonality."""
    required = FEATURE_COLUMNS + ["sellingprice"]
    df = pd.read_csv(features_path, usecols=required).dropna(subset=required)
    df = df[
        df["sellingprice"].between(500, 150_000)
        & df["odometer"].between(1, 500_000)
        & df["vehicle_age"].between(0, 30)
        & df["sale_month"].between(1, 12)
    ].copy()
    df["body"] = df["body"].astype(str).str.lower().str.strip()
    df["sale_month"] = df["sale_month"].astype(int)

    reference_input = df[FEATURE_COLUMNS].copy()
    reference_input["sale_month"] = REFERENCE_MONTH
    reference_input["year_month"] = REFERENCE_YEAR_MONTH
    model = joblib.load(model_path)
    df["reference_prediction"] = np.maximum(model.predict(reference_input), 500.0)

    macro = pd.read_csv(macro_path, usecols=["year_month", "cpi_multiplier"])
    cpi_lookup = macro.set_index("year_month")["cpi_multiplier"]
    df["cpi_multiplier"] = df["year_month"].map(cpi_lookup)
    missing_cpi = sorted(df.loc[df["cpi_multiplier"].isna(), "year_month"].unique())
    if missing_cpi:
        raise ValueError(
            "Missing CPI multipliers for historical sale months: "
            + ", ".join(map(str, missing_cpi))
        )
    df["normalized_price"] = df["sellingprice"] / df["cpi_multiplier"]
    df["price_ratio"] = df["normalized_price"] / df["reference_prediction"]
    return df[
        [
            "body",
            "sale_month",
            "year_month",
            "normalized_price",
            "reference_prediction",
            "price_ratio",
        ]
    ]


def calculate_seasonality_factors(adjusted_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate adjusted rows into shrunken body/month factors."""
    rows = adjusted_rows.copy()
    body_stats = (
        rows.groupby("body", observed=True)
        .agg(body_ratio_median=("price_ratio", "median"), body_count=("price_ratio", "size"))
        .reset_index()
    )
    rows = rows.merge(body_stats[["body", "body_ratio_median"]], on="body", how="left")
    rows["relative_ratio"] = rows["price_ratio"] / rows["body_ratio_median"]
    month_stats = (
        rows.groupby(["body", "sale_month"], observed=True)
        .agg(raw_factor=("relative_ratio", "median"), observations=("relative_ratio", "size"))
        .reset_index()
    )

    output_rows: list[dict] = []
    for body_row in body_stats.itertuples(index=False):
        body_months = month_stats[month_stats["body"] == body_row.body]
        lookup = {int(row.sale_month): row for row in body_months.itertuples(index=False)}
        factors: dict[int, float] = {}
        counts: dict[int, int] = {}
        raw_values: dict[int, float] = {}

        for month in range(1, 13):
            month_row = lookup.get(month)
            if month_row is None:
                raw_factor, count, factor = 1.0, 0, 1.0
            else:
                raw_factor = float(month_row.raw_factor)
                count = int(month_row.observations)
                weight = count / (count + SHRINKAGE_OBSERVATIONS)
                factor = 1.0 + (raw_factor - 1.0) * weight
                factor = float(np.clip(factor, FACTOR_MIN, FACTOR_MAX))
            factors[month] = factor
            counts[month] = count
            raw_values[month] = raw_factor

        eligible_recommendation_months = [
            month for month, count in counts.items()
            if count >= MIN_RECOMMENDATION_OBSERVATIONS
        ]
        has_recommendation = len(eligible_recommendation_months) >= MIN_RECOMMENDATION_MONTHS
        recommendation_months = eligible_recommendation_months.copy()
        if not recommendation_months:
            recommendation_months = [month for month, count in counts.items() if count > 0]
        if not recommendation_months:
            recommendation_months = [REFERENCE_MONTH]
        best_month = max(recommendation_months, key=factors.get)
        worst_month = min(recommendation_months, key=factors.get)

        for month in range(1, 13):
            count = counts[month]
            if count == 0:
                confidence = "no_data"
            elif count < MIN_RECOMMENDATION_OBSERVATIONS:
                confidence = "low"
            elif count < SHRINKAGE_OBSERVATIONS:
                confidence = "medium"
            else:
                confidence = "high"
            output_rows.append(
                {
                    "body": body_row.body,
                    "sale_month": month,
                    "month_name": MONTH_NAMES[month],
                    "seasonal_factor": round(factors[month], 4),
                    "seasonal_delta_pct": round((factors[month] - 1.0) * 100, 1),
                    "raw_factor": round(raw_values[month], 4),
                    "observations": count,
                    "confidence": confidence,
                    "is_observed": count > 0,
                    "body_observations": int(body_row.body_count),
                    "recommendation_eligible_months": len(eligible_recommendation_months),
                    "has_recommendation": has_recommendation,
                    "best_month": best_month,
                    "best_month_name": MONTH_NAMES[best_month],
                    "best_factor": round(factors[best_month], 4),
                    "worst_month": worst_month,
                    "worst_month_name": MONTH_NAMES[worst_month],
                    "worst_factor": round(factors[worst_month], 4),
                }
            )

    return pd.DataFrame(output_rows).sort_values(["body", "sale_month"]).reset_index(drop=True)


def build_seasonality_factors(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    macro_path: Path = MACRO_PATH,
    output_path: Path = SEASONALITY_PATH,
) -> pd.DataFrame:
    """Build and save conservative seasonal factors."""
    adjusted_rows = prepare_seasonality_data(features_path, model_path, macro_path)
    factors = calculate_seasonality_factors(adjusted_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(output_path, index=False)
    return factors


def load_seasonality_factors(path: Path = SEASONALITY_PATH) -> pd.DataFrame:
    """Load seasonality factors and normalize lookup columns."""
    if not path.exists():
        return build_seasonality_factors(output_path=path)
    df = pd.read_csv(path)
    df["body"] = df["body"].astype(str).str.lower().str.strip()
    df["sale_month"] = df["sale_month"].astype(int)
    if "recommendation_eligible_months" not in df.columns:
        eligible = (
            df["observations"].ge(MIN_RECOMMENDATION_OBSERVATIONS)
            .groupby(df["body"])
            .transform("sum")
        )
        df["recommendation_eligible_months"] = eligible.astype(int)
    if "has_recommendation" not in df.columns:
        df["has_recommendation"] = (
            df["recommendation_eligible_months"] >= MIN_RECOMMENDATION_MONTHS
        )
    return df


def get_seasonality_row(body: str, sale_month: int, factors: pd.DataFrame) -> dict:
    """Return a matching seasonal row, or a neutral no-data fallback."""
    normalized_body = str(body).lower().strip()
    month = int(sale_month)
    if month not in MONTH_NAMES:
        raise ValueError(f"sale_month must be between 1 and 12, got {sale_month!r}")
    match = factors[(factors["body"] == normalized_body) & (factors["sale_month"] == month)]
    if not match.empty:
        return match.iloc[0].to_dict()

    return {
        "body": normalized_body,
        "sale_month": month,
        "month_name": MONTH_NAMES.get(month, str(month)),
        "seasonal_factor": 1.0,
        "seasonal_delta_pct": 0.0,
        "observations": 0,
        "confidence": "no_data",
        "is_observed": False,
        "recommendation_eligible_months": 0,
        "has_recommendation": False,
        "best_month": month,
        "best_month_name": MONTH_NAMES.get(month, str(month)),
        "best_factor": 1.0,
        "worst_month": month,
        "worst_month_name": MONTH_NAMES.get(month, str(month)),
        "worst_factor": 1.0,
    }


def get_seasonal_factor(body: str, sale_month: int, factors: pd.DataFrame) -> float:
    return float(get_seasonality_row(body, sale_month, factors)["seasonal_factor"])


def apply_stage3(
    stage2_price: float,
    body: str,
    sale_month: int,
    factors: pd.DataFrame,
) -> tuple[float, float, dict]:
    """Return final price, seasonal factor, and supporting factor row."""
    row = get_seasonality_row(body, sale_month, factors)
    factor = float(row["seasonal_factor"])
    return stage2_price * factor, factor, row
