# Aktueller Stand — Team MAIL (BIS5522)

> **Für KI-Assistenten:** Diese Datei zuerst lesen. Sie ist der zentrale Kontext für den aktuellen Projektstand, getroffene Entscheidungen und nächste Schritte.
> Zuletzt aktualisiert: 2026-06-24

---

## Projekt in 30 Sekunden

**BIS5522 AI & Machine Learning — HS Pforzheim, SoSe 2026**
**Team MAIL:** Johanna Thiele · Moritz Binder · Pascal Müller · Tara Golle
**Deadline:** 31.07.2026 (LNCS-Paper, 12 Seiten + Präsentation)

Wir bauen einen **hybriden KI-Agenten für dynamische Gebrauchtwagenpreisgestaltung**. Das System kombiniert ein ML-Modell (Stage 1), eine makroökonomische CPI-Anpassung (Stage 2) und saisonale Regeln (Stage 3) zu einem marktaktuellen Preisvorschlag.

**Trainingsdaten:** 558.743 US-Auktionsverkäufe 2014–2015 (Manheim via Kaggle).
**Endprodukt:** Streamlit-Demo + LLM-Orchestrierung.

---

## Update vom 24.06.2026 — Stage 2 und Stage 3 vollständig geprüft

- **Stage 2:** Die Evaluation verwendet eine feste Stage-1-Referenz (`2015-02`) und trennt damit Basispreis und CPI-Anpassung sauber. Der Rückwärtstest verbessert den MAE von **1.890,21 $ auf 1.889,19 $**.
- **Stage 3:** Alle **529.790** auswertbaren Verkäufe, **45** Karosserieformen und **540** Kombinationen aus Karosserieform und Monat wurden erneut validiert.
- Der Stage-3-Test-MAE verbessert sich von **1.895,03 $ auf 1.870,20 $** (**−24,82 $ / −1,31 %**).
- Saisonfaktoren bleiben auf **0,85 bis 1,15** begrenzt. Für August bis November fehlen historische Verkäufe; diese Monate bleiben deshalb neutral bei **1,0** und werden als `no_data` gekennzeichnet.
- Eine Empfehlung für den besten bzw. schwächsten Verkaufsmonat wird nur noch ausgegeben, wenn mindestens **zwei Monate mit jeweils 100 Beobachtungen** verfügbar sind. Das trifft auf **20 von 45** Karosserieformen zu; bei den übrigen **25** zeigt die App transparent „Keine belastbare Empfehlung“.
- Fehlende historische CPI-Werte und ungültige Verkaufsmonate werden nun ausdrücklich abgefangen, statt stillschweigend einen neutralen Wert zu verwenden.
- Die aktualisierten Präsentations- und Sprechtextfassungen liegen unter `outputs/`. Die vorherigen Fassungen bleiben dort unter ihren bisherigen Dateinamen als Backup erhalten.

---

## Architektur und Status

```
Eingabe: Fahrzeugbeschreibung (Marke, Modell, Karosserie, Baujahr, Km, Zustand)
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 1 — Micro (FERTIG ✅)                                           │
│  HistGradientBoostingRegressor auf Fahrzeugattribute                   │
│  → Basispreis in USD (2015er Preisniveau)                              │
│  Modell: models/price_model.joblib                                     │
│  MAE: $1.850 | RMSE: $3.299 | R²: 0.882 | MAPE: 16,4%               │
└───────────────────────┬────────────────────────────────────────────────┘
                        │  × cpi_multiplier(Zieldatum)
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — Macro (FERTIG ✅)                                           │
│  CPI Gebrauchtwagen (FRED: CUSR0000SETA01), Basis 2015 = 1,000        │
│  Aktueller Multiplikator (2026-06): 1,2177 → +21,8% vs. 2015         │
│  Rückwärtstest: MAE 1.890,21 → 1.889,19 $ (feste Referenz 2015-02)   │
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
  build_features.py         ← Feature Engineering → car_prices_features.csv
  train_price_model.py      ← Stage-1-Training (HistGB + Segment-Analyse).
                              Speichert models/price_model.joblib.
  stage2_macro.py           ← Stage-2-Modul. Von App und Scripts importieren.
                              Funktionen: load_macro_index(), apply_stage2(),
                              get_cpi_multiplier(), get_macro_context()
  stage3_seasonality.py     ← Stage-3-Modul. Funktionen:
                              prepare_seasonality_data(),
                              build_seasonality_factors(), apply_stage3()
  evaluate_stage2.py        ← Backtest + Vorwärtsprojektion für Stage 2.
                              Schreibt models/stage2_evaluation.json
  evaluate_stage3.py        ← Saisonalitätsfaktoren + Summary.
                              Schreibt models/seasonality_factors.csv,
                              models/stage3_evaluation.json und
                              model_results_stage3.md
  evaluate_segments.py      ← Segmentanalyse auf gespeichertem Modell.
  compare_models.py         ← 6-Modell-Benchmark (Ergebnisse in model_comparison/)
  enrich_macro.py           ← FRED-Download → macro_index.csv (Internet nötig)
  train_stage1.py           ← Ältere XGBoost-Pipeline (nicht primär, Vergleich)

models/
  price_model.joblib        ← Produktions-Stage-1-Modell (HistGradientBoosting)
  price_model_metrics.json  ← Stage-1-Metriken inkl. Segmentaufschlüsselung
  stage2_evaluation.json    ← Stage-2-Backtest + Projektion
  stage1_xgboost.json       ← Älteres XGBoost-Modell (nicht primär)
  stage1_encoder.pkl        ← Encoder für ältere Pipeline

model_comparison/
  model_comparison.md       ← Vergleich 6 Modelle (lesbar)
  model_comparison.json     ← Rohdaten des Benchmarks

model_results.md            ← Stage-1-Ergebnisse (auto-generiert beim Training)
model_results_stage2.md     ← Stage-2-Evaluierung (auto-generiert)
model_results_stage3.md     ← Stage-3-Saisonalität (auto-generiert)
                              inkl. Datenabdeckung und 80/20-Prüfung

docs/
  project_proposal_v2.md   ← Offizieller Projektvorschlag
  data_cleaning.md          ← Datenbereinigungs-Entscheidungen
  feature_engineering.md    ← Feature-Engineering-Entscheidungen
  session_2026-06-04.md     ← Session-Notizen: Macro-Pipeline-Setup
  session_2026-06-09.md     ← Session-Notizen: macro_index-Korrekturen

macro_index.csv             ← FRED-Makrodaten 1996–2026-06 (9 Spalten)
car_prices_clean.csv        ← Bereinigte Auktionsdaten (558.743 Zeilen)
car_prices_features.csv     ← Feature-Engineering-Ergebnis (534.318 Zeilen)
```

> **Nicht im Git (gitignored):**
> `car_prices_macro.csv` (98 MB) — neu erstellen mit `uv run python scripts/enrich_macro.py`

---

## Befehle zum Ausführen

```bash
# Abhängigkeiten installieren (Python 3.12+, uv erforderlich)
uv sync

# 1. Feature-Datensatz erstellen
uv run python scripts/build_features.py

# 2. Stage-1-Modell trainieren (schnell: 200k Zeilen; --max-rows 0 für alle 534k)
uv run python scripts/train_price_model.py

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
| Stage-2-CPI-Mult. (2026-06) | 1,2177 | +21,8% vs. 2015-Basis |
| Stage-2-Backtest | $1.890,21 → $1.889,19 | feste Stage-1-Referenz 2015-02; angezeigtes Δ −$1,02 |
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
| 2 | **Volles Training** | `--max-rows 0` für 534k Zeilen; erwartete MAE-Verbesserung ~5–10% |
| 3 | **LLM-Orchestrierung** | Stage 1–3-Output + `get_macro_context()` → natürlichsprachliche Erklärung |
| 4 | **Paper schreiben** | LNCS 12 Seiten; Architekturdiagramm; Stage-1+2+3-Ergebnisse sind fertig |

---

## Wichtige Hinweise für KI-Assistenten

- `car_prices_macro.csv` ist **gitignored** (98 MB). Bei Bedarf: `uv run python scripts/enrich_macro.py`
- `macro_index.csv` enthält 1996-01 bis 2026-06. Die letzten 3 Monate sind forward-gefüllt (FRED-Verzögerung).
- `models/price_model.joblib` nutzt `OrdinalEncoder(unknown_value=-1)` für Kategorien. `year_month`-Werte außerhalb 2014–2015 werden als -1 kodiert — vertretbar, da `year_month` sehr geringe Feature-Importance hat.
- `stage2_macro.py` nutzt absolute Pfade (`PROJECT_ROOT = Path(__file__).resolve().parent.parent`). Importierbar aus `scripts/` und `app/` (die App macht `sys.path.insert(0, str(PROJECT_ROOT / "scripts"))`).
- **PR #1 (GitHub Classroom) nicht anfassen** — wird automatisch vom Professor-System gepflegt.
- **PR #2** (`Mail_project_moritz` → `main`) ist der aktive Entwicklungs-PR.

---

## Datensätze

| Datensatz | Quelle | Zeilen | Zeitraum |
|---|---|---|---|
| `car_prices_clean.csv` | Manheim via Kaggle | 558.743 | 2014–2015 |
| `car_prices_features.csv` | Feature Engineering | 534.318 | 2014–2015 |
| `macro_index.csv` | FRED (7 Serien) | 366 Monate | 1996–2026 |

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
