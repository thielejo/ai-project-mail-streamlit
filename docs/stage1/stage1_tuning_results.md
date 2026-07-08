# Stage 1 — Tuning (CatBoost + Hubraum)

> Vollständiges Tuning mit Skripten: `tuning/` (README + 5 Schritte).
> Bestes Modell (`price_model_catboost_tuned.cbm`, 107 MB) ist gitignored,
> reproduzierbar via `tuning/04_final_full_dataset.py`.

## Ausgangspunkt

CatBoost + Hubraum (ungetunt): MAE $1.120 (voller Datensatz, Test 9).

## Vorgehen & Ergebnisse

Getunt auf 150k-Stichprobe (4-fache CV), Finale auf vollem Datensatz (529k).

| Schritt | Frage | Ergebnis |
|---|---|---|
| 1 | Zielgröße/Loss? | **Log-Ziel + MAE-Loss** gewinnt ($1.225 vs raw $1.239 vs log+RMSE $1.277) |
| 2 | Hyperparameter (Optuna, 29 Trials) | bester Validierungs-MAE $1.200; depth=10, lr≈0,06, l2≈3,4 … |
| 3 | Abgeleitete Features / Monotonie? | Abgeleitete Features helfen **nicht**; globale Monotonie **schadet** (+112 %) |
| 4 | **Finale (voller Datensatz)** | **MAE $1.042 / R² 0,961 / MAPE 11,8 %** (−6,9 % vs ungetunt) |
| 5 | Gezielte Monotonie (condition/odometer) | **schadet ebenfalls** (+66 %) → verworfen |

## Kernaussagen

- **Bestes Stage-1-Modell: MAE $1.042** (getuntes CatBoost + Hubraum). Größter
  Einzelgewinn kam aus der **Log-Zielgröße**, nicht aus dem HPO.
- **Segment-MAPE:** Mid-Range 6,3 %, Premium 5,9 %, Luxus 8,2 % — im Kernmarkt sehr gut.
- **Zwei ehrliche Negativbefunde:** abgeleitete Features und Monotonie-Constraints
  brachten keinen Nutzen bzw. schadeten. Sauber getestet und mit Begründung verworfen.
- **Monotonie-Detail (Oldtimer):** Eine „älter → billiger"-Regel wäre falsch, weil
  Oldtimer im Preis wieder steigen (an den Daten geprüft: U-Kurve bei 26+ Jahren,
  64 Fahrzeuge). Deshalb wurde `vehicle_age` von vornherein von Constraints
  ausgenommen — und die verbleibenden Constraints ganz verworfen.

## Grenze der Vorhersagbarkeit

$1.042 entsprechen ~7,5 % des Durchschnittspreises (~$13.900). Die $1.000-Marke
ist mit diesen Daten nicht erreichbar: Der verbleibende Fehler ist teils
**irreduzibles Auktionsrauschen** (derselbe Wagen bringt an verschiedenen Tagen
verschiedene Preise) und teils Folge **fehlender Merkmale** (Unfallhistorie,
Ausstattung, Scheckheft), die der Manheim-Datensatz nicht enthält.

## Status

Tuning abgeschlossen und dokumentiert. Das getunte Modell ist **noch nicht**
in die Streamlit-App / Stage 2–3 integriert (die App nutzt weiterhin V2). Bei
einer späteren Integration müssen Stage 2 (CPI-Backtest) und Stage 3
(Saisonfaktoren) gegen das neue Stage-1 neu evaluiert werden — Logik unverändert,
nur die Zahlen.
