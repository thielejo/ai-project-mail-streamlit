"""Generate and evaluate Stage-3 seasonal adjustment factors."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage3_seasonality import (
    FACTOR_MAX,
    FACTOR_MIN,
    MIN_RECOMMENDATION_OBSERVATIONS,
    REFERENCE_YEAR_MONTH,
    SHRINKAGE_OBSERVATIONS,
    calculate_seasonality_factors,
    prepare_seasonality_data,
)

OUTPUT_JSON = Path("models/stage3_evaluation.json")
OUTPUT_MD = Path("model_results_stage3.md")
SEASONALITY_CSV = Path("models/seasonality_factors.csv")
RANDOM_STATE = 42


def _top_body_summary(factors: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    return (
        factors.groupby("body", observed=True)
        .agg(
            total_observations=("body_observations", "first"),
            observed_months=("is_observed", "sum"),
            best_month=("best_month_name", "first"),
            best_factor=("best_factor", "first"),
            worst_month=("worst_month_name", "first"),
            worst_factor=("worst_factor", "first"),
        )
        .reset_index()
        .sort_values("total_observations", ascending=False)
        .head(top_n)
    )


def _holdout_evaluation(adjusted_rows: pd.DataFrame) -> dict[str, float | int]:
    rng = np.random.default_rng(RANDOM_STATE)
    training_mask = rng.random(len(adjusted_rows)) < 0.8
    training_rows = adjusted_rows.loc[training_mask]
    test_rows = adjusted_rows.loc[~training_mask].copy()
    training_factors = calculate_seasonality_factors(training_rows)
    lookup = training_factors.set_index(["body", "sale_month"])["seasonal_factor"]
    test_rows["seasonal_factor"] = [
        float(lookup.get((body, month), 1.0))
        for body, month in zip(test_rows["body"], test_rows["sale_month"])
    ]
    baseline_error = (
        test_rows["normalized_price"] - test_rows["reference_prediction"]
    ).abs()
    seasonal_error = (
        test_rows["normalized_price"]
        - test_rows["reference_prediction"] * test_rows["seasonal_factor"]
    ).abs()
    baseline_mae = float(baseline_error.mean())
    seasonal_mae = float(seasonal_error.mean())
    return {
        "training_rows": int(training_mask.sum()),
        "test_rows": int((~training_mask).sum()),
        "baseline_mae": round(baseline_mae, 2),
        "stage3_mae": round(seasonal_mae, 2),
        "mae_change": round(seasonal_mae - baseline_mae, 2),
        "mae_change_percent": round((seasonal_mae / baseline_mae - 1.0) * 100, 2),
    }


def _month_table_for_body(factors: pd.DataFrame, body: str) -> str:
    rows = factors[factors["body"] == body].sort_values("sale_month")
    return "\n".join(
        f"| {row.month_name} | {row.seasonal_factor:.4f} | "
        f"{row.seasonal_delta_pct:+.1f}% | {int(row.observations):,} | {row.confidence} |"
        for row in rows.itertuples(index=False)
    )


def _write_markdown(
    factors: pd.DataFrame,
    body_summary: pd.DataFrame,
    holdout: dict[str, float | int],
    observed_months: list[int],
) -> None:
    summary_table = "\n".join(
        f"| {row.body} | {int(row.total_observations):,} | {int(row.observed_months)} | "
        f"{row.best_month} | {(row.best_factor - 1.0) * 100:+.1f}% | "
        f"{row.worst_month} | {(row.worst_factor - 1.0) * 100:+.1f}% |"
        for row in body_summary.itertuples(index=False)
    )
    examples = [
        body for body in ["convertible", "suv", "sedan", "coupe"]
        if body in set(factors["body"])
    ]
    details = "\n\n".join(
        f"### {body}\n\n"
        "| Monat | Faktor | Effekt | Beobachtungen | Sicherheit |\n"
        "|---|---:|---:|---:|---|\n"
        f"{_month_table_for_body(factors, body)}"
        for body in examples
    )
    observed_text = ", ".join(str(month) for month in observed_months)
    content = f"""# Stage 3 Evaluation: Seasonal Adjustment

## Methode

Stage 3 ergänzt den Stage-2-Marktpreis um einen saisonalen Faktor nach
Karosserieform und Verkaufsmonat:

```
final_price = stage2_price x seasonal_factor(body, month)
```

Die Berechnung vergleicht CPI-normalisierte Verkaufspreise mit Vorhersagen des
Stage-1-Modells für die feste Referenz `{REFERENCE_YEAR_MONTH}`. Dadurch werden
Unterschiede im Fahrzeugmix (Modell, Alter, Laufleistung und Zustand) weitgehend
herausgerechnet. Der Zielmonat fließt nicht doppelt in Stage 1 und Stage 3 ein.

Pro Karosserieform werden die monatlichen Medianabweichungen relativ zum
Gesamtmedian berechnet. Alle Effekte werden mit einer Prior-Stärke von
{SHRINKAGE_OBSERVATIONS:,} Beobachtungen Richtung 1.0 gedämpft und auf
{FACTOR_MIN:.2f} bis {FACTOR_MAX:.2f} begrenzt.

## Datenabdeckung

- Beobachtete Verkaufsmonate: {observed_text}
- August bis November fehlen im Datensatz vollständig und erhalten deshalb
  neutral den Faktor 1.0. Für diese Monate wird keine Empfehlung behauptet.
- Ein Monat wird nur ab {MIN_RECOMMENDATION_OBSERVATIONS} Beobachtungen als
  bester oder schwächster Verkaufsmonat berücksichtigt.

## Getrennte 80/20-Prüfung der Saisonregel

| Kennzahl | Ohne Stage 3 | Mit Stage 3 | Änderung |
|---|---:|---:|---:|
| MAE auf CPI-normalisierten Preisen | ${holdout['baseline_mae']:,.2f} | ${holdout['stage3_mae']:,.2f} | {holdout['mae_change_percent']:+.2f}% |

Die Faktoren wurden dabei nur aus den 80% Regel-Trainingsdaten abgeleitet und
auf den übrigen {holdout['test_rows']:,} Zeilen geprüft. Dies ist eine Prüfung
der Saisonregel, kein vollständig unabhängiger neuer Stage-1-Modelltest.

## Wichtigste Muster

| Karosserie | Beobachtungen | Monate mit Daten | Bester Monat | Effekt | Schwächster Monat | Effekt |
|---|---:|---:|---|---:|---|---:|
{summary_table}

## Beispiel-Faktoren

{details}

## Einordnung

- Die korrigierten Effekte sind deutlich kleiner als beim Vergleich roher
  Monatspreise. Das ist plausibel, weil teurere oder jüngere Fahrzeuge in
  einzelnen Monaten nun nicht mehr als Saisonalität fehlinterpretiert werden.
- Die Daten stammen fast vollständig aus Dezember 2014 bis Juli 2015. Stage 3
  bleibt daher eine konservative Heuristik, kein kausaler Nachweis.
"""
    OUTPUT_MD.write_text(content, encoding="utf-8")


def main() -> None:
    print("Stage 3 Evaluation")
    print("==================")
    print("1. Bereite CPI- und fahrzeugmixbereinigte Daten vor...")
    adjusted_rows = prepare_seasonality_data()
    print(f"   -> {len(adjusted_rows):,} verwertbare Verkäufe")

    print("2. Berechne und speichere saisonale Faktoren...")
    factors = calculate_seasonality_factors(adjusted_rows)
    SEASONALITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(SEASONALITY_CSV, index=False)
    body_summary = _top_body_summary(factors)

    print("3. Prüfe Saisonregel auf getrennten 20% der Daten...")
    holdout = _holdout_evaluation(adjusted_rows)
    print(
        f"   -> MAE ${holdout['baseline_mae']:,.2f} auf ${holdout['stage3_mae']:,.2f} "
        f"({holdout['mae_change_percent']:+.2f}%)"
    )
    observed_months = sorted(adjusted_rows["sale_month"].unique().astype(int).tolist())
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "cpi_normalized_model_residual_by_body_month",
        "reference_year_month": REFERENCE_YEAR_MONTH,
        "factor_min": FACTOR_MIN,
        "factor_max": FACTOR_MAX,
        "shrinkage_observations": SHRINKAGE_OBSERVATIONS,
        "minimum_recommendation_observations": MIN_RECOMMENDATION_OBSERVATIONS,
        "observed_months": observed_months,
        "unobserved_months": [month for month in range(1, 13) if month not in observed_months],
        "holdout_evaluation": holdout,
        "seasonality_csv": str(SEASONALITY_CSV),
        "body_summary": body_summary.to_dict(orient="records"),
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_markdown(factors, body_summary, holdout, observed_months)
    print(f"4. Ergebnisse gespeichert: {SEASONALITY_CSV}, {OUTPUT_JSON}, {OUTPUT_MD}")


if __name__ == "__main__":
    main()
