# Stage 1 — CatBoost-Experiment (Kandidat)

> **Status:** Experiment / Kandidat — noch nicht Produktionsmodell.
> Experiment-Code liegt im Branch `Mail_project_moritz` unter
> `vin_fin_enrichment/08_catboost_vs_best.py`.

## Motivation

Bei tabellarischen Daten mit vielen hochkardinalen kategorialen Merkmalen
(`trim`, `state`, `make_model`, `color`, `interior` …) behandelt **CatBoost**
diese Kategorien nativ und schlägt One-Hot-basierte Verfahren oft mit weniger
Tuning. Wir haben das gegen unser bestes bisheriges Modell (V2 + Hubraum) getestet.

## Ergebnis (voller Datensatz, 529.169 Zeilen, identischer Split rs=42, identische Features)

| Modell | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Schlankes Stage 1 (HistGB) | $1.824 | $3.253 | 0,885 | 16,3 % |
| V2 (XGBoost-Ensemble, reiche Merkmale) | $1.365 | $2.390 | 0,937 | 15,0 % |
| V2 + Hubraum | $1.201 | $2.040 | 0,954 | 13,7 % |
| **CatBoost + Hubraum** | **$1.120** | **$2.035** | **0,954** | **12,6 %** |

→ CatBoost senkt den MAE um **−$81 (−6,8 %)** gegenüber V2 + Hubraum und hat die
niedrigste MAPE (12,6 %).

## Warum das attraktiv ist

- **Besser:** niedrigster MAE und MAPE aller bisher getesteten Stage-1-Varianten.
- **Einfacher:** ein **einzelnes** Modell statt des Zwei-Modell-Voting-Ensembles
  von V2 — weniger Komplexität, leichter zu warten und zu erklären.
- **Kaum getunt:** Standard-Konfiguration (2000 Iterationen, depth 8, lr 0,05,
  MAE-Loss, native kategoriale Merkmale). Mit Hyperparameter-Tuning ist weitere
  Verbesserung wahrscheinlich.

## Methode

- Daten/Features/Filter/Split **identisch** zum V2-Vergleich (fairer A/B).
- Features: model_year, vehicle_age, odometer, condition, **displacement** (FIN/Hubraum)
  + make, model, trim, body, transmission, state, color, interior, make_model.
- CatBoost mit `loss_function="MAE"`, kategoriale Merkmale nativ via `cat_features`.

## Einschränkungen / offen

- Bewertung auf **einem** Train/Test-Split. Für eine belastbare, zitierfähige
  Aussage steht eine **Kreuzvalidierung** (z. B. 5-fach) noch aus.
- **Hyperparameter-Tuning** (z. B. mit Optuna) ist als nächster Schritt geplant
  und dürfte den Wert weiter verbessern.
- Noch **nicht** in die Streamlit-App / Stage 2–3 integriert. Bei einem späteren
  Wechsel des Produktions-Stage-1 müssen Stage 2 (CPI-Backtest) und Stage 3
  (Saisonfaktoren aus Stage-1-Residuen) gegen das neue Modell **neu evaluiert**
  werden — die Logik bleibt unverändert, nur die Zahlen.

## Reproduktion

```bash
# Branch Mail_project_moritz, Voraussetzung: car_prices_fin.csv (clean + Hubraum)
uv run python vin_fin_enrichment/08_catboost_vs_best.py
```
