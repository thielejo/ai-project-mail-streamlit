"""Train V1 and V2 from scratch on one shared, untouched train/test split.

This is the strict Stage-1 comparison. Both models receive exactly the same
rows for training and exactly the same rows for testing. Existing saved model
files are not loaded or overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from train_price_model import (
    FEATURE_COLUMNS as V1_FEATURES,
    build_sklearn_pipeline as build_v1_model,
)
from train_stage1_v2 import (
    FEATURES as V2_FEATURES,
    TARGET,
    build_model as build_v2_model,
    load_data,
    metrics,
    segment_metrics,
)

INPUT_PATH = Path("car_prices_clean.csv")
OUTPUT_JSON = Path("models/stage1_v1_v2_shared_split.json")
OUTPUT_MD = Path("docs/stage1/model_results_stage1_v1_v2_shared_split.md")
RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    return parser.parse_args()


def split_fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def paired_bootstrap_ci(
    v1_absolute_error: np.ndarray,
    v2_absolute_error: np.ndarray,
) -> dict[str, float | int]:
    """Bootstrap the paired per-row MAE improvement (V1 error minus V2 error)."""
    difference = v1_absolute_error - v2_absolute_error
    rng = np.random.default_rng(RANDOM_STATE)
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for sample in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(difference), size=len(difference))
        estimates[sample] = float(difference[indices].mean())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "mean_mae_improvement_dollars": round(float(difference.mean()), 2),
        "ci95_low_dollars": round(float(low), 2),
        "ci95_high_dollars": round(float(high), 2),
        "v2_lower_error_row_percent": round(float((difference > 0).mean() * 100), 2),
        "equal_error_row_percent": round(float((difference == 0).mean() * 100), 2),
    }


def write_markdown(payload: dict, path: Path) -> None:
    v1 = payload["v1_metrics"]
    v2 = payload["v2_metrics"]
    comparison = payload["comparison"]
    paired = payload["paired_analysis"]
    rows = [
        "# Stage 1: Strenger V1/V2-Vergleich auf gemeinsamem Split",
        "",
        "## Versuchsaufbau",
        "",
        "V1 und V2 wurden in diesem Lauf von Grund auf neu trainiert. Beide Modelle",
        "erhielten exakt dieselben Trainingszeilen und wurden anschließend auf exakt",
        "denselben, zuvor unangetasteten Testzeilen bewertet. Gespeicherte Modelle",
        "wurden nicht geladen und nicht überschrieben.",
        "",
        f"- Bereinigte Zeilen: {payload['rows_used']:,}",
        f"- Trainingszeilen: {payload['train_rows']:,}",
        f"- Testzeilen: {payload['test_rows']:,}",
        f"- Split: 80/20 mit `random_state={payload['random_state']}`",
        f"- Testsplit-Fingerabdruck: `{payload['test_split_sha256']}`",
        "",
        "## Ergebnis",
        "",
        "| Modell | MAE | RMSE | R² | MAPE |",
        "|---|---:|---:|---:|---:|",
        f"| V1 – neu trainierter HistGradientBoostingRegressor | ${v1['mae']:,.2f} | ${v1['rmse']:,.2f} | {v1['r2']:.4f} | {v1['mape_percent']:.2f}% |",
        f"| V2 – neu trainiertes XGBoost-Ensemble | ${v2['mae']:,.2f} | ${v2['rmse']:,.2f} | {v2['r2']:.4f} | {v2['mape_percent']:.2f}% |",
        "",
        f"- MAE-Verbesserung: **${comparison['mae_improvement_dollars']:,.2f}**",
        f"- Relative MAE-Verbesserung: **{comparison['mae_improvement_percent']:.2f}%**",
        f"- 95%-Bootstrap-Intervall der MAE-Verbesserung: **${paired['ci95_low_dollars']:,.2f} bis ${paired['ci95_high_dollars']:,.2f}**",
        f"- V2 hat auf {paired['v2_lower_error_row_percent']:.2f}% der einzelnen Testzeilen den kleineren absoluten Fehler.",
        "",
        "## Interpretation",
        "",
        "Dieser Vergleich beseitigt die Unsicherheit über eine mögliche Überschneidung",
        "zwischen dem früheren V1-Training und dem V2-Testset. Die ausgewiesene",
        "Verbesserung ist deshalb der methodisch bevorzugte V1/V2-Wert.",
        "",
        "## Einschränkungen",
        "",
        "- Der Split ist zufällig und kein zeitlicher Zukunftstest.",
        "- Beide Modelle nutzen nur Zeilen, auf denen alle für V2 benötigten Merkmale vorhanden sind.",
        "- Die Daten stammen aus US-Auktionen und überwiegend aus 2014–2015.",
        "- V1 und V2 unterscheiden sich gleichzeitig in Modellarchitektur und Merkmalen; der Test misst den Gesamteffekt des V2-Upgrades.",
        "",
        "## Reproduktion",
        "",
        "```powershell",
        "uv run python scripts/compare_stage1_v1_v2_shared_split.py --max-rows 0",
        "```",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    print("Lade gemeinsam bereinigte Daten...")
    data = load_data(args.input, args.max_rows)
    all_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    print(f"Zeilen: {len(data):,}; Training: {len(train):,}; Test: {len(test):,}")

    print("Trainiere V1 auf dem gemeinsamen Trainingssplit...")
    v1_model = build_v1_model()
    v1_model.fit(train[V1_FEATURES], train[TARGET])
    v1_prediction = np.maximum(v1_model.predict(test[V1_FEATURES]), 0)

    print("Trainiere V2 auf demselben Trainingssplit...")
    v2_model = build_v2_model()
    v2_model.fit(train[V2_FEATURES], train[TARGET])
    v2_prediction = np.maximum(v2_model.predict(test[V2_FEATURES]), 0)

    y_test = test[TARGET]
    v1_metrics = metrics(y_test, v1_prediction)
    v2_metrics = metrics(y_test, v2_prediction)
    improvement = v1_metrics["mae"] - v2_metrics["mae"]
    improvement_percent = improvement / v1_metrics["mae"] * 100
    paired = paired_bootstrap_ci(
        np.abs(y_test.to_numpy() - v1_prediction),
        np.abs(y_test.to_numpy() - v2_prediction),
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "both_models_retrained_on_one_shared_untouched_split",
        "input_path": str(args.input),
        "rows_used": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "random_state": RANDOM_STATE,
        "test_split_sha256": split_fingerprint(test_indices),
        "v1": {
            "model": "HistGradientBoostingRegressor with log target",
            "features": V1_FEATURES,
        },
        "v2": {
            "model": "50/50 XGBoost raw/log ensemble",
            "features": V2_FEATURES,
        },
        "v1_metrics": v1_metrics,
        "v2_metrics": v2_metrics,
        "comparison": {
            "mae_improvement_dollars": round(float(improvement), 2),
            "mae_improvement_percent": round(float(improvement_percent), 2),
        },
        "paired_analysis": paired,
        "v1_segment_metrics": segment_metrics(y_test, v1_prediction),
        "v2_segment_metrics": segment_metrics(y_test, v2_prediction),
        "limitations": [
            "Random split, not a temporal future holdout.",
            "Shared row set requires all V2 features to be present.",
            "Measures the combined effect of architecture and feature changes.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, args.output_md)

    print(json.dumps({
        "v1_mae": v1_metrics["mae"],
        "v2_mae": v2_metrics["mae"],
        "improvement_dollars": payload["comparison"]["mae_improvement_dollars"],
        "improvement_percent": payload["comparison"]["mae_improvement_percent"],
        "ci95": [paired["ci95_low_dollars"], paired["ci95_high_dollars"]],
    }, indent=2))
    print(f"Gespeichert: {args.output_json}, {args.output_md}")


if __name__ == "__main__":
    main()
