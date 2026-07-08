# Test 7 — Bestes Stage-1 (HistGB) vs. + FIN (voller Datensatz)

Zeilen nach Filter: **529,790** | Test: 105,958 | Modell: HistGradientBoosting (Produktions-Hyperparameter)

## Gesamtergebnis

| Arm | MAE | RMSE | R² | MAPE | Δ MAE vs. A |
|---|---:|---:|---:|---:|---:|
| A) Baseline (Produktion) | $1,824 | $3,253 | 0.8848 | 16.3% |  |
| B) + Hubraum | $1,584 | $2,724 | 0.9192 | 15.1% | −$240 (+13.1 %) |
| C) + alle FIN | $1,592 | $2,722 | 0.9194 | 15.1% | −$232 (+12.7 %) |

## Segmentfehler — Arm B (+ Hubraum)

| Segment | n | MAE | MAPE |
|---|---:|---:|---:|
| Budget | 17,698 | $768 | 35.0% |
| Economy | 21,617 | $1,122 | 15.1% |
| Mid-Range | 46,152 | $1,376 | 9.6% |
| Premium | 20,518 | $2,662 | 10.1% |
| Luxury | 1,814 | $8,119 | 14.3% |
