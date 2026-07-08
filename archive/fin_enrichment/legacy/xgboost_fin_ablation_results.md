# Test 8 (FINAL) — Pascals V2 vs. V2 + Hubraum (voller Datensatz)

Zeilen nach Filter: **529,169** | Test: 105,834 | Modell: XGBoost-Voting-Ensemble (exakt V2)

| Arm | MAE | RMSE | R² | MAPE | Δ MAE vs. V2 |
|---|---:|---:|---:|---:|---:|
| A) V2 (reiche Merkmale) | $1,365 | $2,390 | 0.9371 | 15.0% |  |
| B) V2 + Hubraum | $1,199 | $2,038 | 0.9543 | 13.7% | −$166 (+12.2 %) |

## Segmentfehler — Arm B (V2 + Hubraum)

| Segment | n | MAE | MAPE |
|---|---:|---:|---:|
| Budget | 17,773 | $793 | 38.4% |
| Economy | 21,689 | $993 | 13.5% |
| Mid-Range | 46,049 | $1,045 | 7.4% |
| Premium | 20,475 | $1,765 | 6.8% |
| Luxury | 1,780 | $5,187 | 9.1% |
