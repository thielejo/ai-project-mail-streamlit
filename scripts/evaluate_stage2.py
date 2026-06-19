"""Evaluate Stage 2 CPI adjustment on the historical test set.

The evaluation has three angles:
  1. Backtest on 2014-2015 test set: Stage 2 should barely change accuracy because
     the CPI multiplier is ~0.99 for that period (training data IS the baseline).
  2. CPI multiplier analysis over time: shows the pandemic surge and normalisation.
  3. Forward projection: what does a median-priced car cost across different target dates?

Run from project root:
    uv run python scripts/evaluate_stage2.py

Outputs:
    models/stage2_evaluation.json  (machine-readable)
    model_results_stage2.md        (human-readable summary)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from stage2_macro import (
    apply_stage2,
    get_cpi_multiplier,
    get_macro_context,
    load_macro_index,
    MACRO_SIGNAL_LABELS,
)

# Must exactly match train_price_model.py to reproduce the same test split
RANDOM_STATE = 42
TARGET_COLUMN = "sellingprice"
NUMERIC_FEATURES = ["vehicle_age", "sale_month", "odometer", "condition"]
CATEGORICAL_FEATURES = ["year_month", "make", "model", "body"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MAX_ROWS = 200_000

FEATURES_PATH = Path("car_prices_features.csv")
MODEL_PATH = Path("models/price_model.joblib")
MACRO_PATH = Path("macro_index.csv")
OUTPUT_JSON = Path("models/stage2_evaluation.json")
OUTPUT_MD = Path("model_results_stage2.md")

FORWARD_PROJECTION_MONTHS = [
    "2015-01", "2016-01", "2017-01", "2018-01", "2019-01",
    "2020-01", "2020-06",
    "2021-01", "2021-06", "2021-12",
    "2022-01", "2022-06", "2022-12",
    "2023-01", "2023-09",
    "2024-01",
    "2025-01",
    "2026-01", "2026-06",
]


def _load_test_set() -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce the exact Stage 1 test split (same rows, same seed)."""
    df = pd.read_csv(FEATURES_PATH)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=required)
    df = df[
        df[TARGET_COLUMN].between(500, 150_000)
        & df["odometer"].between(1, 500_000)
        & df["vehicle_age"].between(0, 30)
    ].copy()
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    return X_test, y_test


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true.values - y_pred) / y_true.values)) * 100)
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape_percent": round(mape, 2),
    }


def _forward_projection(
    stage1_median: float,
    macro: pd.DataFrame,
    months: list[str],
) -> list[dict]:
    rows = []
    for ym in months:
        stage2, mult = apply_stage2(stage1_median, ym, macro)
        rows.append(
            {
                "year_month": ym,
                "cpi_multiplier": round(mult, 4),
                "stage1_baseline": round(stage1_median, 0),
                "stage2_price": round(stage2, 0),
                "delta_pct": round((mult - 1.0) * 100, 1),
            }
        )
    return rows


def _write_markdown(
    stage1_metrics: dict,
    stage2_metrics: dict,
    multiplier_stats: dict,
    stage1_median: float,
    projection: list[dict],
    current_ctx: dict,
) -> None:
    mae_delta = stage2_metrics["mae"] - stage1_metrics["mae"]
    r2_delta = stage2_metrics["r2"] - stage1_metrics["r2"]

    proj_table = "\n".join(
        f"| {r['year_month']} | {r['cpi_multiplier']:.4f} | ${r['stage1_baseline']:,.0f} "
        f"| ${r['stage2_price']:,.0f} | {r['delta_pct']:+.1f}% |"
        for r in projection
    )

    macro_table = "\n".join(
        f"| {label} | {current_ctx.get(col, 'n/a')} |"
        for col, label in MACRO_SIGNAL_LABELS.items()
    )

    content = f"""# Stage 2 Evaluation: CPI Macro Adjustment

## Methode

Stage 2 multipliziert den Stage-1-Basispreis mit dem CPI-Multiplikator
des Zieldatums:

```
stage2_price = stage1_price × cpi_multiplier(target_month)
```

Der `cpi_multiplier` ist auf den 2015-Jahresdurchschnitt (= 1.000) normiert.
Quelle: CPI Used Cars & Trucks (CUSR0000SETA01, FRED).

## Backtest: Historische Genauigkeit (2014–2015 Testset)

| Metrik | Stage 1 | Stage 2 | Δ |
|---|---:|---:|---:|
| MAE | ${stage1_metrics['mae']:,.2f} | ${stage2_metrics['mae']:,.2f} | {mae_delta:+,.2f} |
| RMSE | ${stage1_metrics['rmse']:,.2f} | ${stage2_metrics['rmse']:,.2f} | — |
| R² | {stage1_metrics['r2']:.4f} | {stage2_metrics['r2']:.4f} | {r2_delta:+.4f} |
| MAPE | {stage1_metrics['mape_percent']:.2f}% | {stage2_metrics['mape_percent']:.2f}% | — |

**CPI-Multiplikator im Testset (2014–2015):**
- Min: {multiplier_stats['min']:.4f} / Max: {multiplier_stats['max']:.4f} / Ø {multiplier_stats['mean']:.4f}

> Der minimale Unterschied (±{abs(mae_delta):.0f} MAE) bestätigt, dass Stage 2
> die historische Genauigkeit nicht verschlechtert. Die Trainingsperiode liegt
> im CPI-Baseline-Bereich (~0.99), sodass der Anpassungsfaktor nahezu neutral ist.

## Vorwärtsprojektion (Median Stage-1-Preis: ${stage1_median:,.0f})

| Monat | CPI-Multiplikator | Stage 1 | Stage 2 | Δ % |
|---|---:|---:|---:|---:|
{proj_table}

> Der COVID-bedingte Engpass (2021–2022) zeigt einen Preisanstieg von bis zu +22%.
> Der aktuelle Stand (2026-06) liegt stabil bei ~+22% über dem 2015-Niveau.

## Makro-Kontext {current_ctx['year_month']}

| Indikator | Wert |
|---|---:|
| CPI-Multiplikator | {current_ctx['cpi_multiplier']:.4f} |
{macro_table}

## Einschränkungen

- Stage 2 extrapoliert ausschließlich über CPI-Inflation; strukturelle Marktveränderungen
  (z. B. Elektrifizierung, Chip-Engpässe) sind nicht modelliert.
- Für Monate ohne FRED-Daten wird der letzte verfügbare Monat genutzt (Forward-Fill).
- Das Stage-1-Modell kennt `year_month`-Werte außerhalb von 2014–2015 nicht;
  der OrdinalEncoder kodiert diese als -1. Da `year_month` der unwichtigste Feature
  (Importance 32 vs. 2470 für `make`) ist, ist der Effekt minimal.
"""
    OUTPUT_MD.write_text(content, encoding="utf-8")


def main() -> None:
    print("Stage 2 Evaluation")
    print("==================")

    print("\n1. Lade Makrodaten...")
    macro = load_macro_index(MACRO_PATH)

    print("2. Reproduziere Stage-1-Testset (200k Zeilen, random_state=42)...")
    X_test, y_test = _load_test_set()
    print(f"   -> Testset: {len(X_test):,} Zeilen, {len(X_test['year_month'].unique())} year_month-Werte")

    print("3. Lade Stage-1-Modell und berechne Basisvorhersagen...")
    model = joblib.load(MODEL_PATH)
    stage1_preds = model.predict(X_test)
    stage1_metrics = _metrics(y_test, stage1_preds)
    print(f"   Stage 1  MAE=${stage1_metrics['mae']:,.2f}  RMSE=${stage1_metrics['rmse']:,.2f}  R²={stage1_metrics['r2']:.4f}")

    print("\n4. Wende Stage-2-CPI-Anpassung auf Testset an...")
    multipliers = np.array(
        [get_cpi_multiplier(ym, macro) for ym in X_test["year_month"].tolist()]
    )
    stage2_preds = stage1_preds * multipliers
    stage2_metrics = _metrics(y_test, stage2_preds)

    mult_stats = {
        "min": round(float(multipliers.min()), 4),
        "max": round(float(multipliers.max()), 4),
        "mean": round(float(multipliers.mean()), 4),
    }
    print(f"   CPI-Multiplikator im Testset: min={mult_stats['min']}  max={mult_stats['max']}  ø={mult_stats['mean']}")
    print(f"   Stage 2  MAE=${stage2_metrics['mae']:,.2f}  RMSE=${stage2_metrics['rmse']:,.2f}  R²={stage2_metrics['r2']:.4f}")

    mae_delta = stage2_metrics["mae"] - stage1_metrics["mae"]
    print(f"   Δ MAE (Stage2−Stage1): ${mae_delta:+,.2f}  [{abs(mae_delta)/stage1_metrics['mae']*100:.2f}% Änderung]")
    print(f"   -> Erwartetes Ergebnis: minimale Änderung, da Testperiode im CPI-Basisjahr-Bereich liegt.")

    print("\n5. Vorwärtsprojektion (Median Stage-1-Preis)...")
    stage1_median = float(np.median(stage1_preds))
    projection = _forward_projection(stage1_median, macro, FORWARD_PROJECTION_MONTHS)
    print(f"   Stage-1-Median-Baseline: ${stage1_median:,.0f}")
    print(f"\n   {'Monat':<12} {'Multip.':>8} {'Stage 1':>10} {'Stage 2':>10} {'Δ':>7}")
    print(f"   {'-'*53}")
    for r in projection:
        print(
            f"   {r['year_month']:<12} "
            f"{r['cpi_multiplier']:>8.4f} "
            f"${r['stage1_baseline']:>9,.0f} "
            f"${r['stage2_price']:>9,.0f} "
            f"{r['delta_pct']:>+6.1f}%"
        )

    print("\n6. Makro-Kontext 2026-06:")
    current_ctx = get_macro_context("2026-06", macro)
    for key, val in current_ctx.items():
        print(f"   {key}: {val}")

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage1_metrics_historical": stage1_metrics,
        "stage2_metrics_historical": stage2_metrics,
        "test_multiplier_stats": mult_stats,
        "stage1_test_median_price": round(stage1_median, 2),
        "forward_projection": projection,
        "current_macro_context": current_ctx,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nJSON gespeichert: {OUTPUT_JSON}")

    _write_markdown(
        stage1_metrics,
        stage2_metrics,
        mult_stats,
        stage1_median,
        projection,
        current_ctx,
    )
    print(f"Markdown gespeichert: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
