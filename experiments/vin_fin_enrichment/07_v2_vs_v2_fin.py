"""
Test 8 (FINAL) — Pascals echtes V2 vs. V2 + Hubraum, voller Datensatz.

Repliziert EXAKT das V2-Ensemble aus scripts/train_stage1_v2.py (main):
XGBoost VotingRegressor 50/50 (reg:squarederror roh + reg:absoluteerror log),
OneHotEncoder(min_frequency=20), 700 Baeume, gleiche Features/Filter/Split.

Frage: Bringt der Hubraum (FIN) noch Zusatzwert ZUSAETZLICH zu V2 (das bereits
trim, color, interior, transmission, state nutzt)?

   A) V2            = model_year, vehicle_age, odometer, condition +
                      make, model, trim, body, transmission, state, color,
                      interior, make_model
   B) V2 + Hubraum  = A + displacement

Datenquelle: car_prices_fin.csv (clean + FIN). Voller Datensatz.

Aufruf:
    uv run python experiments/vin_fin_enrichment/07_v2_vs_v2_fin.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "car_prices_fin.csv"
OUT_MD = Path(__file__).resolve().parent / "v2_vs_v2_fin_results.md"
RANDOM_STATE = 42
TARGET = "sellingprice"

NUMERIC = ["model_year", "vehicle_age", "odometer", "condition"]
CATEGORICAL = ["make", "model", "trim", "body", "transmission",
               "state", "color", "interior", "make_model"]

PRICE_SEGMENTS = [("Budget", 500, 5_000), ("Economy", 5_000, 10_000),
                  ("Mid-Range", 10_000, 20_000), ("Premium", 20_000, 40_000),
                  ("Luxury", 40_000, 150_000)]


def load() -> pd.DataFrame:
    cols = ["year", "saledate", "make", "model", "trim", "body", "transmission",
            "state", "condition", "odometer", "color", "interior", TARGET, "displacement"]
    df = pd.read_csv(SRC, usecols=cols)
    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    df = df.dropna(subset=[c for c in cols if c != "displacement"]).copy()
    df["model_year"] = pd.to_numeric(df["year"], errors="coerce")
    df["vehicle_age"] = (df["saledate"].dt.year - df["model_year"]).clip(lower=0)
    df = df[df[TARGET].between(500, 150_000)
            & df["odometer"].between(1, 500_000)
            & df["vehicle_age"].between(0, 30)
            & df["condition"].between(1, 5)].copy()
    for c in CATEGORICAL:
        if c == "make_model":
            continue
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["make_model"] = df["make"] + "|" + df["model"]
    df["displacement"] = pd.to_numeric(df["displacement"], errors="coerce")
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())
    return df.reset_index(drop=True)


def build_xgb(objective: str) -> XGBRegressor:
    return XGBRegressor(objective=objective, eval_metric="mae", n_estimators=700,
                        learning_rate=0.045, max_depth=9, min_child_weight=12,
                        subsample=0.90, colsample_bytree=0.90, reg_alpha=0.02,
                        reg_lambda=1.0, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)


def build_model(num, cat) -> VotingRegressor:
    def pre():
        return ColumnTransformer([
            ("numeric", "passthrough", num),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True), cat),
        ])
    raw = Pipeline([("preprocessor", pre()), ("model", build_xgb("reg:squarederror"))])
    logp = TransformedTargetRegressor(
        regressor=Pipeline([("preprocessor", pre()), ("model", build_xgb("reg:absoluteerror"))]),
        func=np.log1p, inverse_func=np.expm1)
    return VotingRegressor(estimators=[("raw_price", raw), ("log_price", logp)], weights=[0.5, 0.5])


def metr(y, p) -> dict:
    return {"mae": mean_absolute_error(y, p), "rmse": np.sqrt(mean_squared_error(y, p)),
            "r2": r2_score(y, p), "mape": np.mean(np.abs((y - p) / y)) * 100}


def main() -> None:
    df = load()
    print(f"Zeilen nach Filter: {len(df):,}", flush=True)
    y = df[TARGET]
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    y_test = y.iloc[te]

    arms = {"A) V2 (reiche Merkmale)": (NUMERIC, CATEGORICAL),
            "B) V2 + Hubraum": (NUMERIC + ["displacement"], CATEGORICAL)}

    results, seg_b = {}, None
    for label, (num, cat) in arms.items():
        print(f"Trainiere {label} ...", flush=True)
        X = df[num + cat]
        m = build_model(num, cat)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        r = metr(y_test, pred)
        results[label] = r
        print(f"  {label:24} MAE ${r['mae']:,.0f}  RMSE ${r['rmse']:,.0f}  R2 {r['r2']:.4f}  MAPE {r['mape']:.1f}%", flush=True)
        if label.startswith("B"):
            ps = pd.Series(pred, index=y_test.index)
            seg_b = [(n, int(mk.sum()), mean_absolute_error(y_test[mk], ps[mk]),
                      np.mean(np.abs((y_test[mk]-ps[mk])/y_test[mk]))*100)
                     for n, lo, hi in PRICE_SEGMENTS
                     for mk in [y_test.between(lo, hi)] if mk.sum() >= 10]

    base = results["A) V2 (reiche Merkmale)"]["mae"]
    d = base - results["B) V2 + Hubraum"]["mae"]
    print(f"\n=== Zusatznutzen Hubraum ueber V2: -${d:,.0f} ({d/base*100:+.2f} %) ===", flush=True)

    lines = ["# Test 8 (FINAL) — Pascals V2 vs. V2 + Hubraum (voller Datensatz)\n",
             f"Zeilen nach Filter: **{len(df):,}** | Test: {len(te):,} | Modell: XGBoost-Voting-Ensemble (exakt V2)\n",
             "| Arm | MAE | RMSE | R² | MAPE | Δ MAE vs. V2 |", "|---|---:|---:|---:|---:|---:|"]
    for label, r in results.items():
        dd = "" if label.startswith("A") else f"−${base-r['mae']:,.0f} ({(base-r['mae'])/base*100:+.1f} %)"
        lines.append(f"| {label} | ${r['mae']:,.0f} | ${r['rmse']:,.0f} | {r['r2']:.4f} | {r['mape']:.1f}% | {dd} |")
    if seg_b:
        lines += ["\n## Segmentfehler — Arm B (V2 + Hubraum)\n", "| Segment | n | MAE | MAPE |", "|---|---:|---:|---:|"]
        for n, cnt, mae, mape in seg_b:
            lines.append(f"| {n} | {cnt:,} | ${mae:,.0f} | {mape:.1f}% |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ergebnis geschrieben: {OUT_MD.name}", flush=True)


if __name__ == "__main__":
    main()
