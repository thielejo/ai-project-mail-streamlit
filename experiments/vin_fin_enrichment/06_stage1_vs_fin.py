"""
Test 7 — Bestes Stage-1 (HistGB) vs. Stage-1 + FIN, voller Datensatz.

Repliziert EXAKT die Produktions-Pipeline aus scripts/train_price_model.py
(HistGradientBoosting, gleiche Features, Hyperparameter, log1p, StandardScaler +
OrdinalEncoder, 80/20-Split, identische Filter) und vergleicht drei Arme auf
demselben Split:

   A) Baseline   = vehicle_age, sale_month, odometer, condition,
                   year_month, make, model, body            (= Produktion)
   B) + Hubraum  = A + displacement                          (Kernfeature, Test 6)
   C) + alle FIN = A + displacement, fuel_type, cylinders

Datenquelle: car_prices_fin.csv (clean + FIN). Abgeleitete Features
(vehicle_age, sale_month, year_month) werden wie in build_features.py berechnet.

Aufruf:
    uv run python experiments/vin_fin_enrichment/06_stage1_vs_fin.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "car_prices_fin.csv"
OUT_MD = Path(__file__).resolve().parent / "stage1_vs_fin_results.md"
RANDOM_STATE = 42

BASE_NUM = ["vehicle_age", "sale_month", "odometer", "condition"]
BASE_CAT = ["year_month", "make", "model", "body"]
TARGET = "sellingprice"

PRICE_SEGMENTS = [
    ("Budget", 500, 5_000), ("Economy", 5_000, 10_000),
    ("Mid-Range", 10_000, 20_000), ("Premium", 20_000, 40_000),
    ("Luxury", 40_000, 150_000),
]


def load() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    df = df.dropna(subset=["year", "make", "model", "body", "condition",
                           "odometer", "saledate", "sellingprice"]).copy()
    df["sale_year"] = df["saledate"].dt.year
    df["sale_month"] = df["saledate"].dt.month
    df["year_month"] = df["saledate"].dt.strftime("%Y-%m")
    df["vehicle_age"] = (df["sale_year"] - df["year"]).clip(lower=0)
    # gleiche Filter wie train_price_model.load_modeling_data
    df = df[df[TARGET].between(500, 150_000)
            & df["odometer"].between(1, 500_000)
            & df["vehicle_age"].between(0, 30)].copy()
    # FIN aufbereiten
    df["displacement"] = pd.to_numeric(df["displacement"], errors="coerce")
    df["cylinders"] = pd.to_numeric(df["cylinders"], errors="coerce")
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())
    df["cylinders"] = df["cylinders"].fillna(df["cylinders"].median())
    df["fuel_type"] = df["fuel_type"].fillna("unknown")
    return df


def make_model() -> TransformedTargetRegressor:
    """Exakt wie build_sklearn_pipeline() in train_price_model.py."""
    def build(num, cat):
        pre = ColumnTransformer([
            ("numeric", StandardScaler(), num),
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1), cat),
        ])
        reg = HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=0.05, random_state=RANDOM_STATE)
        pipe = Pipeline([("preprocessor", pre), ("model", reg)])
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return build


def metrics(y, p) -> dict:
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": np.sqrt(mean_squared_error(y, p)),
        "r2": r2_score(y, p),
        "mape": np.mean(np.abs((y - p) / y)) * 100,
    }


def main() -> None:
    df = load()
    print(f"Zeilen nach Filter: {len(df):,}", flush=True)
    build = make_model()
    y = df[TARGET]
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    y_test = y.iloc[te]

    arms = {
        "A) Baseline (Produktion)": (BASE_NUM, BASE_CAT),
        "B) + Hubraum": (BASE_NUM + ["displacement"], BASE_CAT),
        "C) + alle FIN": (BASE_NUM + ["displacement", "cylinders"], BASE_CAT + ["fuel_type"]),
    }

    results = {}
    seg_b = None
    for label, (num, cat) in arms.items():
        X = df[num + cat]
        m = build(num, cat)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        res = metrics(y_test, pred)
        results[label] = res
        print(f"{label:26} MAE ${res['mae']:,.0f}  RMSE ${res['rmse']:,.0f}  "
              f"R2 {res['r2']:.4f}  MAPE {res['mape']:.1f}%", flush=True)
        if label.startswith("B"):
            # Segmentfehler fuer den Hubraum-Arm
            ps = pd.Series(pred, index=y_test.index)
            seg_b = []
            for name, lo, hi in PRICE_SEGMENTS:
                mask = y_test.between(lo, hi)
                if mask.sum() >= 10:
                    seg_b.append((name, int(mask.sum()),
                                  mean_absolute_error(y_test[mask], ps[mask]),
                                  np.mean(np.abs((y_test[mask]-ps[mask])/y_test[mask]))*100))

    base = results["A) Baseline (Produktion)"]["mae"]
    print("\n=== Verbesserung vs. Baseline A ===")
    for label, r in results.items():
        if label.startswith("A"):
            continue
        d = base - r["mae"]
        print(f"  {label:18}: -${d:,.0f}  ({d/base*100:+.2f} %)")

    # Markdown schreiben
    lines = ["# Test 7 — Bestes Stage-1 (HistGB) vs. + FIN (voller Datensatz)\n",
             f"Zeilen nach Filter: **{len(df):,}** | Test: {len(te):,} | Modell: HistGradientBoosting (Produktions-Hyperparameter)\n",
             "## Gesamtergebnis\n",
             "| Arm | MAE | RMSE | R² | MAPE | Δ MAE vs. A |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, r in results.items():
        d = "" if label.startswith("A") else f"−${base-r['mae']:,.0f} ({(base-r['mae'])/base*100:+.1f} %)"
        lines.append(f"| {label} | ${r['mae']:,.0f} | ${r['rmse']:,.0f} | {r['r2']:.4f} | {r['mape']:.1f}% | {d} |")
    if seg_b:
        lines += ["\n## Segmentfehler — Arm B (+ Hubraum)\n",
                  "| Segment | n | MAE | MAPE |", "|---|---:|---:|---:|"]
        for name, n, mae, mape in seg_b:
            lines.append(f"| {name} | {n:,} | ${mae:,.0f} | {mape:.1f}% |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nErgebnis geschrieben: {OUT_MD.name}")


if __name__ == "__main__":
    main()
