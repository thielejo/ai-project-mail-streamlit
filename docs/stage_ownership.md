# Stage Ownership und Projektlogik

Diese Datei hält fest, wohin Änderungen fachlich gehören. Sie verhindert, dass
Stage-3-Arbeit versehentlich als neues Basispreis-Modell verstanden wird.

## Stage 1 — Fahrzeugwert / Micro Model

Stage 1 schätzt den zeitneutralen Fahrzeug-Basiswert aus Fahrzeugmerkmalen.

Hierhin gehören:

- Training und Auswahl des Basispreis-Modells
- V1/V2-Vergleich
- XGBoost-/HistGB-Modellarchitektur
- Fahrzeugmerkmale wie Marke, Modell, Ausstattung, Getriebe, Bundesstaat,
  Außenfarbe, Innenfarbe, Kilometerstand und Zustand
- `make_model`-Interaktion
- Modellmetriken wie MAE, RMSE, R² und MAPE
- Runtime-Logik, die das Stage-1-Modell lädt und Eingaben in Modellfeatures übersetzt

Wichtige Dateien:

- `scripts/train_price_model.py` — ursprüngliches Stage-1-Modell
- `scripts/train_stage1_v2.py` — Stage-1-V2-Modell
- `scripts/optimize_stage1_v2.py` — Stage-1-V2-Optimierung
- `scripts/compare_stage1_v1_v2_shared_split.py` — strenger V1/V2-Vergleich
- `scripts/stage1_runtime.py` — Stage-1-Laden und Input-Building für die App
- `models/price_model.joblib` — V1-Fallback
- `models/price_model_v2.joblib` — Stage-1-V2-Produktionsmodell
- `docs/stage1/` — Stage-1-Ergebnisse und V2-Übergabe

Wichtig: Die 25,17-%-Verbesserung gehört zu Stage 1, nicht zu Stage 3.

## Stage 2 — Marktpreisniveau / Macro

Stage 2 passt den Stage-1-Basispreis an das Preisniveau eines Zielmonats an.

Hierhin gehören:

- CPI-Gebrauchtwagenindex
- CPI-Multiplikator relativ zum 2015-Basisniveau
- Forward-Fill für noch nicht veröffentlichte Makromonate
- Backtests und Projektionen der CPI-Korrektur

Wichtige Dateien:

- `scripts/stage2_macro.py`
- `scripts/enrich_macro.py`
- `scripts/evaluate_stage2.py`
- `macro_index.csv`
- `models/stage2_evaluation.json`
- `docs/stage2/model_results_stage2.md`

Stage 2 ist kein neues Fahrzeugmodell. Es multipliziert den Stage-1-Wert mit
dem Marktpreisfaktor.

## Stage 3 — Saisonalität

Stage 3 ist ausschließlich die saisonale Feinkorrektur nach Karosserieform und
Monat.

Hierhin gehören:

- Saisonfaktoren nach `body` und `sale_month`
- Best-/Worst-Month-Logik
- neutrale Faktoren bei fehlender oder dünner Datenbasis
- Evaluation der saisonalen Regel

Wichtige Dateien:

- `scripts/stage3_seasonality.py`
- `scripts/evaluate_stage3.py`
- `models/seasonality_factors.csv` — ursprüngliche V1-nahe Faktoren
- `models/seasonality_factors_v2.csv` — aktuelle Faktoren auf Basis von Stage 1 V2
- `models/stage3_evaluation.json`
- `docs/stage3/model_results_stage3.md`

Stage 3 baut kein neues Modell. Sie nutzt die Residuen des Stage-1-Modells, um
vorsichtige saisonale Faktoren zu schätzen.

## App / Demo / UX

Die Streamlit-App kombiniert Stage 1 bis 3 und macht sie für Nutzer bedienbar.

Hierhin gehören:

- deutsche UI-Begriffe
- Kilometer-Eingabe und interne Meilenumrechnung
- Sterne-Skala für Zustand
- Dropdown-Validierung, z. B. Karosserieform passend zu Marke und Modell
- Entwicklerdetails hinter Toggles
- Anzeige von Preisaufbau, Modellvergleich und saisonalen Hinweisen

Wichtige Datei:

- `app/streamlit_app.py`

Die App ist kein eigener fachlicher Stage-Baustein, sondern die Integration der
drei Stages.

## Archiv

Historische oder missverständlich benannte Übergaben liegen unter `archive/`.
Diese Dateien bleiben nachvollziehbar, sind aber nicht mehr die primäre
Projektlogik.
