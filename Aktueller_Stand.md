# Aktueller Stand — Team MAIL (BIS5522)

> **Für KI-Assistenten:** Diese Datei zuerst lesen. Sie ist der zentrale Kontext für den aktuellen Projektstand, getroffene Entscheidungen und nächste Schritte.
> Zuletzt aktualisiert: 2026-06-28

---

## Projekt in 30 Sekunden

**BIS5522 AI & Machine Learning — HS Pforzheim, SoSe 2026**
**Team MAIL:** Johanna Thiele · Moritz Binder · Pascal Müller · Tara Golle
**Deadline:** 31.07.2026 (LNCS-Paper, 12 Seiten + Präsentation)

Wir bauen einen **hybriden KI-Agenten für dynamische Gebrauchtwagenpreisgestaltung**. Das System kombiniert ein ML-Modell (Stage 1), eine makroökonomische CPI-Anpassung (Stage 2) und saisonale Regeln (Stage 3) zu einem marktaktuellen Preisvorschlag.

**Trainingsdaten:** 558.743 US-Auktionsverkäufe 2014–2015 (Manheim via Kaggle).
**Endprodukt:** Streamlit-Demo + LLM-Orchestrierung.

---

## Update vom 28.06.2026 — Stages fachlich neu sortiert

Die aktuelle Implementierung ist jetzt wieder am Projektplan ausgerichtet:

- **Stage 1:** Fahrzeugwert-Modell. Hierhin gehört das neue V2-XGBoost-Ensemble inklusive zusätzlicher Fahrzeugmerkmale und V1/V2-Vergleich.
- **Stage 2:** CPI-Marktpreisanpassung. Hierhin gehört ausschließlich das Marktpreisniveau über den Gebrauchtwagen-CPI.
- **Stage 3:** Saisonalität. Hierhin gehört ausschließlich die saisonale Feinkorrektur nach Karosserieform und Monat.
- **App / UX:** Streamlit-Oberfläche, deutsche Labels, Kilometer-Umrechnung, Sterne-Skala, Toggle-Bereiche und Eingabevalidierung.

Die frühere Stage-Ownership-Notiz liegt inzwischen im Archiv.
Alte, missverständlich benannte Übergaben wurden nach `archive/` verschoben.

### Stage 1 — Neues V2-Fahrzeugwertmodell

- Zusätzlich zum ersten HistGradientBoosting-Modell wurde ein separates **Stage-1-XGBoost-Ensemble** entwickelt. Das ältere Modell liegt inzwischen als Legacy-Artefakt im Archiv.
- V2 ist ein **50/50-Ensemble aus zwei XGBoost-Modellen**: Eine Komponente prognostiziert den Preis direkt in Dollar, die andere den logarithmierten Preis. Architektur, Hyperparameter und Gewichtung wurden auf einem separaten Validierungsanteil gewählt.
- V2 verwendet 529.169 bereinigte Verkäufe sowie zusätzliche Fahrzeugmerkmale wie Ausstattungsvariante, Getriebe, Bundesstaat, Außen- und Innenfarbe. Eine explizite Marke-Modell-Interaktion verbessert die Abbildung verschiedener Modellreihen.
- `MMR`, VIN und Verkäufer wurden bewusst ausgeschlossen. Insbesondere `MMR` wäre bereits eine externe Preisvorhersage und könnte die Modellleistung künstlich beziehungsweise zielähnlich verbessern.
- Im strengen gemeinsamen Neutraining auf exakt demselben Split sinkt der MAE von V1 mit **1.830,95 $** auf **1.370,15 $** bei V2. Das entspricht **460,80 $ beziehungsweise 25,17 % weniger MAE**. Das gepaarte 95%-Bootstrap-Intervall liegt bei **450,74 $ bis 470,83 $**.
- Weitere V2-Testwerte: **RMSE 2.400,34 $**, **R² 0,9366**, **MAPE 15,13 %**. Das Ensemble ist gezielt auf den MAE in Dollar optimiert; andere Fehlermaße können gegenüber einer einzelnen V2-Komponente einen Trade-off zeigen.
- Das aktuelle Produktionsmodell liegt unter `models/stage1_production_model.joblib`. Die App bietet die zusätzlichen Eingaben an; das ältere Modell bleibt als archivierter Fallback erhalten.
- Wichtig: Die **25,17-%-Verbesserung gehört zu Stage 1**, nicht zu Stage 3.

### Stage 1 — FIN-Erweiterung (Hubraum)

- Aus der VIN (US-Pendant zur FIN) lässt sich über die kostenlose NHTSA-API der **Hubraum** nachladen (Abdeckung 99 %). Die aktuelle FIN-/VIN-Vorbereitung liegt unter `vin_fin_enrichment/`; ältere Voruntersuchungen liegen im Archiv.
- Ergänzt man V2 um den Hubraum, sinkt der MAE auf dem vollen Datensatz (529.169 Zeilen) von **1.365 $ (V2) auf 1.201 $** — **−12 %**, R² 0,937 → **0,954**. Damit ist es der **beste Stage-1-Wert** des Projekts.
- Der Hubraum trägt den gesamten FIN-Effekt; Kraftstoff/Zylinder sind redundant (Per-Feature-Ablation, Test 6). Der Effekt hält auch zusätzlich zu trim/Farbe/Ausstattung — also kein Overlap.
- Integriertes Modell: `models/stage1_fin_model.joblib`, Skript `scripts/train_stage1_production_fin.py` (= XGBoost-Pipeline + `displacement`).
- Status: trainiert und dokumentiert; **noch nicht** in die Streamlit-App eingebunden (das Produktionsmodell bleibt vorerst `stage1_production_model.joblib`). Der 47-MB-Voll-Decode-Cache ist nicht im Repo (regenerierbar via `vin_fin_enrichment/build_full_vin_cache.py`).

### Stage 2 — CPI-Marktpreisanpassung

- Stage 2 wurde auf dem vollständigen V2-Testset neu geprüft: **MAE 1.370,16 $ vor CPI und 1.376,22 $ nach CPI** im basisnahen Zeitraum 2014–2015 (+0,44 %).
- Stage 2 verändert die historische Genauigkeit nur leicht, weil die Testdaten nahe am CPI-Basisjahr 2015 liegen.
- Der eigentliche Nutzen von Stage 2 liegt in der Projektion auf spätere Marktpreisniveaus, z. B. 2021–2026.

### Stage 3 — Saisonale Feinkorrektur

- Stage 3 wurde mit dem zeitneutralen Stage-1-V2-Basiswert neu berechnet.
- Auf dem getrennten Regel-Holdout sinkt der MAE von **1.353,15 $ auf 1.339,84 $** (−0,98 %).
- Saisonfaktoren bleiben auf **0,85 bis 1,15** begrenzt. Für August bis November fehlen historische Verkäufe; diese Monate bleiben deshalb neutral bei **1,0** und werden als `no_data` gekennzeichnet.
- Eine Empfehlung für den besten bzw. schwächsten Verkaufsmonat wird nur ausgegeben, wenn mindestens **zwei Monate mit jeweils 100 Beobachtungen** verfügbar sind. Das trifft auf **20 von 45** Karosserieformen zu.
- Stage 3 baut **kein neues Modell**. Sie ist nur die Saisonregel auf Basis der bereinigten Stage-1-Abweichungen.

---

## Architektur und Status

```
Eingabe: Fahrzeugbeschreibung (Marke, Modell, Karosserie, Baujahr, Km, Zustand)
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 1 — Micro (FERTIG ✅)                                           │
│  V2 XGBoost-Raw/Log-Ensemble auf Fahrzeugattribute                    │
│  → Basispreis in USD (2015er Preisniveau)                              │
│  Modell: models/stage1_production_model.joblib                         │
│  MAE: $1.370 | RMSE: $2.400 | R²: 0.9366 | MAPE: 15,13%             │
└───────────────────────┬────────────────────────────────────────────────┘
                        │  × cpi_multiplier(Zieldatum)
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — Macro (FERTIG ✅)                                           │
│  CPI Gebrauchtwagen (FRED: CUSR0000SETA01), Basis 2015 = 1,000        │
│  Aktueller Multiplikator (2026-06): 1,2177 → +21,8% vs. 2015         │
│  V2-Rückwärtstest: MAE 1.370,16 → 1.376,22 $ (+0,44%)                │
│  Modul: scripts/stage2_macro.py                                        │
│  Evaluation: scripts/evaluate_stage2.py                                │
└───────────────────────┬────────────────────────────────────────────────┘
                        │  × seasonal_factor(Karosserietyp, Monat)
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 3 — Saisonal (FERTIG ✅)                                        │
│  Regelbasiert: bereinigte Modellabweichung nach Karosserie × Monat   │
│  → „Bester Monat zum Verkaufen" + saisonaler Anpassungsfaktor         │
│  → Empfehlung nur bei ≥2 Monaten mit jeweils ≥100 Beobachtungen       │
└───────────────────────┬────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LLM-Orchestrierung (OFFEN ❌)                                         │
│  Kombiniert Stage 1–3 + Makrokontext zu einer natürlichsprachlichen   │
│  Erklärung. Die Makrosignale sind bereits in get_macro_context()       │
│  in stage2_macro.py hinterlegt.                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Dateiübersicht

```
Aktueller_Stand.md          ← Diese Datei. KI-Kontext, zuerst lesen.
README.md                   ← Menschenlesbare Übersicht (auf Englisch).

app/
  streamlit_app.py          ← Demo-App. Importiert stage2_macro und
                              stage3_seasonality aus scripts/.
                              Zeigt Stage-1-Basis + Stage-2-CPI-Preis
                              + Stage-3-Saisonpreis.
                              Bewertungsdatum-Selector 1996–2026.

scripts/
  build_features.py         ← Feature Engineering → data/car_prices_features.csv
  train_stage1_production.py
                            ← aktuelles Stage-1-Produktionsmodell.
                              Schreibt models/stage1_production_model.joblib.
  train_stage1_fin.py       ← FIN-/Hubraum-Erweiterung für Stage 1.
  stage2_macro.py           ← Stage-2-Modul. Von App und Scripts importieren.
                              Funktionen: load_macro_index(), apply_stage2(),
                              get_cpi_multiplier(), get_macro_context()
  stage3_seasonality.py     ← Stage-3-Modul. Funktionen:
                              prepare_seasonality_data(),
                              build_seasonality_factors(), apply_stage3()
  evaluate_stage2.py        ← Backtest + Vorwärtsprojektion für Stage 2.
                              Schreibt models/stage2_evaluation.json und
                              docs/stage2/model_results_stage2.md
  evaluate_stage3.py        ← Saisonalitätsfaktoren + Summary.
                              Schreibt models/stage3_seasonality_factors.csv,
                              models/stage3_evaluation.json und
                              docs/stage3/model_results_stage3.md
  compare_models.py         ← 6-Modell-Benchmark (Ergebnisse in model_comparison/)
  enrich_macro.py           ← FRED-Download → data/macro_index.csv (Internet nötig)

models/
  stage1_production_model.joblib
                            ← Produktionsmodell für Stage 1
  stage1_production_metrics.json
                            ← Metriken zum Produktionsmodell
  stage1_fin_model.joblib   ← FIN-/Hubraum-Erweiterung als trainiertes Artefakt
  stage1_fin_metrics.json   ← Metriken zur FIN-/Hubraum-Erweiterung
  stage2_evaluation.json    ← Stage-2-Backtest + Projektion
  stage3_evaluation.json    ← Stage-3-Regel-Holdout
  stage3_seasonality_factors.csv
                            ← saisonale Korrekturfaktoren

docs/
  stage1/                   ← V2-Übergabe und Stage-1-V1/V2-Ergebnisse.
  stage2/                   ← CPI-Report.
  stage3/                   ← Saison-Report.

archive/
  model_artifacts_2026-07-08/
                            ← alte Modellartefakte, Fallbacks und Vergleichsoutputs

model_comparison/
  model_comparison.json     ← Rohdaten des Benchmarks

docs/stage1/stage1_current_model.md
                            ← aktuelles Stage-1-Ergebnis
docs/stage1/stage1_model_comparison.md
                            ← Stage-1-Modellvergleich
docs/stage1/stage1_tuning_results.md
                            ← aktuelles Tuning-Ergebnis
docs/stage1/stage1_benchmark_models.md
                            ← Benchmark der getesteten Modellfamilien
docs/stage2/model_results_stage2.md
                            ← Stage-2-Evaluierung
docs/stage3/model_results_stage3.md
                            ← Stage-3-Saisonalität inkl. Datenabdeckung

docs/
  project_proposal_v2.md   ← Offizieller Projektvorschlag
  data_cleaning.md          ← Datenbereinigungs-Entscheidungen
  feature_engineering.md    ← Feature-Engineering-Entscheidungen
  session_2026-06-04.md     ← Session-Notizen: Macro-Pipeline-Setup
  session_2026-06-09.md     ← Session-Notizen: macro_index-Korrekturen

data/macro_index.csv             ← FRED-Makrodaten 1996–2026-06 (9 Spalten)
data/car_prices_clean.csv        ← Bereinigte Auktionsdaten (558.743 Zeilen)
data/car_prices_features.csv     ← Feature-Engineering-Ergebnis (534.318 Zeilen)
```

> **Nicht im Git (gitignored):**
> `data/car_prices_macro.csv` (98 MB) — neu erstellen mit `uv run python scripts/enrich_macro.py`

---

## Befehle zum Ausführen

```bash
# Abhängigkeiten installieren (Python 3.12+, uv erforderlich)
uv sync

# 1. Feature-Datensatz erstellen
uv run python scripts/build_features.py

# 2. Stage-1-Produktionsmodell trainieren (schnell: 200k Zeilen; --max-rows 0 für alle 534k)
uv run python scripts/train_stage1_production.py

# 3. Stage-2-Evaluation
uv run python scripts/evaluate_stage2.py

# 4. Stage-3-Evaluation
uv run python scripts/evaluate_stage3.py

# 5. Demo-App starten
uv run streamlit run app/streamlit_app.py

# Optional: Makrodaten von FRED aktualisieren
uv run python scripts/enrich_macro.py
```

---

## Aktuelle Metriken

| Metrik | Wert | Anmerkung |
|---|---|---|
| Stage-1-MAE | $1.850 | 40k Testzeilen (20%-Split von 200k) |
| Stage-1-RMSE | $3.299 | |
| Stage-1-R² | 0,882 | Median-Baseline R² = −0,025 |
| Stage-1-MAPE | 16,4% | Bestes Segment: Mid-Range 10,7% |
| Stage-1-V2-MAE | $1.370,15 | 105.834 Testzeilen; 25,17% besser als neu trainiertes V1 auf gemeinsamem Split |
| Stage-1-V2-RMSE | $2.400,34 | V2-Ensemble, auf MAE optimiert |
| Stage-1-V2-R² | 0,9366 | altes Modell auf demselben Test: 0,8800 |
| Stage-1-V2-MAPE | 15,13% | ohne MMR, VIN oder Verkäufer |
| Stage-2-CPI-Mult. (2026-06) | 1,2177 | +21,8% vs. 2015-Basis |
| Stage-2-Backtest mit V2 | $1.370,16 → $1.376,22 | CPI-Korrektur im basisnahen Zeitraum 2014–2015; +0,44 % |
| Trainingszeilen | 200.000 | Teilmenge; voller Datensatz: 534.318 |

**Fehler nach Preissegment (Stage 1):**

| Segment | Preisbereich | MAPE |
|---|---|---|
| Budget | $500–$5k | 35% |
| Economy | $5k–$10k | 16% |
| Mid-Range | $10k–$20k | 11% ← bestes Segment |
| Premium | $20k–$40k | 12% |
| Luxury | $40k+ | 21% |

---

## Getroffene Designentscheidungen

**Stage 1: Warum HistGradientBoosting statt XGBoost?**
XGBoost braucht `libomp` auf macOS (häufig fehlend). HistGB ist der automatische Fallback und schneidet im Benchmark besser ab (MAE $1.850 vs. $2.055). Das Trainings-Skript wählt automatisch.

Diese Entscheidung beschreibt das weiterhin vorhandene V1-Fallback. Das Produktionsmodell V2 nutzt ein abgestimmtes XGBoost-Ensemble und zusätzliche Fahrzeugmerkmale.

**Stage 2: Warum nur CPI-Multiplikator, kein Composite-Index?**
`year_month` ist das unwichtigste Feature in Stage 1 (Permutation Importance 32 vs. 2.470 für `make`). Andere Makrosignale (Leitzins, Arbeitslosigkeit, Stimmung) beeinflussen die Nachfrage, bräuchten aber ein eigenes Gewichtungsmodell. CPI Gebrauchtwagen misst direkt die Inflation auf dem relevanten Markt. Die anderen Signale sind über `get_macro_context()` für die LLM-Schicht verfügbar.

**Stage 2: Warum Forward-Fill für fehlende Monate?**
FRED publiziert mit ~1 Monat Verzögerung. `_resolve_year_month()` in `stage2_macro.py` nutzt den letzten verfügbaren Monat — konsistent mit dem Verhalten von `enrich_macro.py`.

**Warum 2015 als Basisjahr?**
Die Trainingsdaten stammen aus 2014–2015. 2015 ist das dominante Trainingsjahr und im Projektvorschlag als neutrales Baseline-Jahr definiert.

**App-UI: Warum ein einheitliches Bewertungsdatum?**
Das alte UI trennte `sale_year` und `model_year` nicht klar. Ein einziges „Bewertungsdatum" (1996–2026) macht das zweistufige Konzept transparent: Stage 1 gibt den 2015er Basiswert, Stage 2 passt ihn auf das gewählte Datum an.

---

## Nächste Schritte (priorisiert)

| Priorität | Aufgabe | Hinweise |
|---|---|---|
| 1 | **Preisrange in App** | ±MAE des jeweiligen Segments statt einer einzigen Zahl anzeigen |
| 2 | **LLM-Orchestrierung** | Stage 1–3-Output + `get_macro_context()` → natürlichsprachliche Erklärung |
| 3 | **Paper schreiben** | LNCS 12 Seiten; Architekturdiagramm; Stage-1+2+3-Ergebnisse sind fertig |
| 4 | **Eingabe-Plausibilisierung in App** | Nur Karosserieformen/Optionen anbieten, die zum gewählten Modell passen (z. B. Modell → erlaubte `body`-Werte aus den Trainingsdaten ableiten); zusätzlich Warnhinweis bei sehr seltener Marke/Modell-Kombination |

> **Bekanntes Problem (Eingabe-Plausibilisierung) — Beispiel für das Limitations-Kapitel:**
> Wählt man in der App `make=lamborghini`, `model=gallardo`, `body=g sedan`, liefert das Modell nur ~$8.300 Basispreis — völlig unrealistisch für einen Supersportwagen.
> Ursachen: (1) **Widersprüchliche Eingabe** — die echten 4 Gallardos im Datensatz sind `convertible`; „g sedan" ist eine **günstige Limousinen-Kategorie** (Ø ~$19.900), der das Modell folgt. (2) **Kaum Datenbasis** — nur 4 Lamborghinis unter 558.743 Zeilen; das Modell kann kein Markenpremium lernen und regrediert zum Mittelwert. (3) **Dokumentierte Luxus-Schwäche** — Segment >$40k hat die höchste MAPE (~21 %); nur 0,085 % aller Fahrzeuge kosten über $80k.
> Die App lässt aktuell **beliebige Marke-Modell-Karosserie-Kombinationen** zu, auch unmögliche. Eine Plausibilisierung (nur valide Kombinationen zulassen) würde solche irreführenden Schätzungen verhindern. Bis dahin: im Paper als bewusste Datenabdeckungs-Grenze benennen (Modell ist für den Massenmarkt $5k–$40k zuverlässig, nicht für Exoten).

---

## Wichtige Hinweise für KI-Assistenten

- `data/car_prices_macro.csv` ist **gitignored** (98 MB). Bei Bedarf: `uv run python scripts/enrich_macro.py`
- `data/macro_index.csv` enthält 1996-01 bis 2026-06. Die letzten 3 Monate sind forward-gefüllt (FRED-Verzögerung).
- `models/stage1_production_model.joblib` ist das produktiv eingebundene Stage-1-Modell. Es erwartet zusätzlich `trim`, `transmission`, `state`, `color`, `interior` und `make_model`. Fehlt die Datei, lädt die App den archivierten Legacy-Fallback.
- `stage2_macro.py` nutzt absolute Pfade (`PROJECT_ROOT = Path(__file__).resolve().parent.parent`). Importierbar aus `scripts/` und `app/` (die App macht `sys.path.insert(0, str(PROJECT_ROOT / "scripts"))`).
- **PR #1 (GitHub Classroom) nicht anfassen** — wird automatisch vom Professor-System gepflegt.
- **PR #2** (`Mail_project_moritz` → `main`) ist der aktive Entwicklungs-PR.

---

## Datensätze

| Datensatz | Quelle | Zeilen | Zeitraum |
|---|---|---|---|
| `data/car_prices_clean.csv` | Manheim via Kaggle | 558.743 | 2014–2015 |
| `data/car_prices_features.csv` | Feature Engineering | 534.318 | 2014–2015 |
| `data/macro_index.csv` | FRED (7 Serien) | 366 Monate | 1996–2026 |

**Genutzte FRED-Serien:**
- `CUSR0000SETA01` — CPI Gebrauchtwagen (primär für Stage 2)
- `FEDFUNDS` — Leitzins
- `UMCSENT` — Konsumentenstimmung (Univ. Michigan)
- `UNRATE` — US-Arbeitslosenquote
- `TOTALSA` — Fahrzeugverkäufe gesamt (SAAR)
- `USREC` — NBER-Rezessionsindikator
- `BAMLH0A0HYM2` — High-Yield-Credit-Spread

---

## Repository und Kontakt

- **GitHub:** `digital-business-lectures/ai-project-mail`
- **Moritz Binder** (@moritzb1) — `itsmoribind@gmail.com`
- **Aktiver Branch:** `Mail_project_moritz`
- **Aktiver PR:** #2 (`Mail_project_moritz` → `main`)
- **Feedback-PR:** #1 (Professor, nicht anfassen)
