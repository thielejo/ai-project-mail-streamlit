# Stage 1 V2 – Integration, Nutzung und KI-Übergabe

> Vollständiger Kontext für Teammitglieder und andere KI-Assistenten.  
> Arbeitsbranch: `mail_project_pascal`  
> Stand: 24.06.2026  
> `main` wurde nicht verändert.

## 1. Projektüberblick

Die App schätzt einen Gebrauchtwagenpreis in drei getrennten Schritten:

1. **Stage 1 – Fahrzeugwert:** Machine Learning bewertet Fahrzeugmerkmale.
2. **Stage 2 – Markt:** Der Basiswert wird mit dem US-Gebrauchtwagen-CPI auf das gewählte Datum angepasst.
3. **Stage 3 – Saison:** Eine kleine Korrektur berücksichtigt Karosserieform und Monat.

```text
Finaler Preis = Stage-1-Basiswert × CPI-Multiplikator × Saisonfaktor
```

V2 ersetzt ausschließlich das Modell in Stage 1. Markt und Saison bleiben nachvollziehbare Korrekturschichten.

## 2. App für Teammitglieder starten

Voraussetzungen: Zugriff auf das private GitHub-Repository, Git, Python 3.12+ und möglichst `uv`.

### Neues Verzeichnis unter Windows

```powershell
git clone https://github.com/digital-business-lectures/ai-project-mail.git
cd ai-project-mail
git switch mail_project_pascal
uv sync
uv run streamlit run app/streamlit_app.py
```

### Bereits vorhandenes Repository

```powershell
cd C:\Users\<BENUTZERNAME>\Documents\GitHub\ai-project-mail
git fetch origin --prune
git switch mail_project_pascal
git pull --ff-only origin mail_project_pascal
uv sync
uv run streamlit run app/streamlit_app.py
```

Ohne `uv`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Danach `http://localhost:8501` öffnen. Das Terminal muss offen bleiben; Beenden mit `Strg+C`.

### macOS/Linux

```bash
git clone https://github.com/digital-business-lectures/ai-project-mail.git
cd ai-project-mail
git switch mail_project_pascal
uv sync
uv run streamlit run app/streamlit_app.py
```

## 3. V1 und V2 im direkten Vergleich

Gespeicherte Auswertung auf denselben 105.834 Fahrzeugzeilen:

| Modell | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| V1 – auf gemeinsamem Split neu trainiert | 1.830,95 $ | 3.276,81 $ | 0,8818 | 16,45 % |
| V2 | 1.370,15 $ | 2.400,34 $ | 0,9366 | 15,13 % |

```text
Absolut: 1.830,95 $ − 1.370,15 $ = 460,80 $
Relativ: 460,80 $ ÷ 1.830,95 $ × 100 = 25,17 %
```

Der MAE ist der durchschnittliche absolute Abstand zwischen Prognose und echtem Verkaufspreis. V2 liegt durchschnittlich rund 461 $ näher am echten Preis. Ein niedrigerer MAE ist besser.

Die **25,17 % beziehen sich nur auf V1 gegen V2 in Stage 1**. Sie stammen nicht aus Stage 2 oder Stage 3.

## 4. Warum V2 genauer ist

### Mehr Daten

- V1-Produktionslauf: 200.000 Zeilen
- V2: 529.169 bereinigte Verkäufe
- V2-Training: 423.335 Zeilen
- V2-Test: 105.834 Zeilen

### Mehr Fahrzeugmerkmale

V1 verwendet:

```text
vehicle_age, sale_month, odometer, condition,
year_month, make, model, body
```

V2 verwendet:

```text
model_year, vehicle_age, odometer, condition,
make, model, trim, body, transmission, state,
color, interior, make_model
```

Neu sind insbesondere Ausstattung, Getriebe, Region, Farben und die Kombination `make_model`. Dadurch kann das Modell verschiedene Baureihen und Ausstattungen genauer unterscheiden.

### Stärkeres Modell

V1 nutzt einen `HistGradientBoostingRegressor`. V2 kombiniert zwei XGBoost-Modelle:

1. Modell auf dem direkten Dollarpreis
2. Modell auf dem logarithmierten Preis
3. 50/50-Mittelwert beider Vorhersagen

Das direkte Modell ist stark bei Dollarfehlern. Das logarithmische Modell behandelt günstige und teure Fahrzeuge ausgewogener. Das Ensemble verbindet beide Perspektiven.

### Kategorien und unbekannte Werte

V2 verwendet One-Hot-Encoding mit `handle_unknown="ignore"` und `min_frequency=20`. Unbekannte Eingaben führen dadurch nicht zum Absturz; sehr seltene Kategorien werden begrenzt.

### Klare Aufgabentrennung

V2 enthält bewusst weder `sale_month` noch `year_month`:

- Stage 1: Fahrzeug
- Stage 2: Marktzeitpunkt
- Stage 3: Saison

So wird der Monat nicht mehrfach berücksichtigt.

## 5. Bewusst ausgeschlossene Merkmale

- **MMR:** bereits eine externe Preisbewertung und zu nah an der Zielvariable; könnte die Leistung künstlich schönrechnen.
- **VIN:** zu fahrzeugspezifisch und schlecht auf unbekannte Fahrzeuge übertragbar.
- **Verkäufer:** verhindert, dass das Modell hauptsächlich Händler wiedererkennt.
- **Verkaufsmonat/Jahr-Monat:** gehören fachlich in Stage 2 und 3.

## 6. Training und Modellarchitektur

- Daten: Manheim US-Auktionen, überwiegend 2014–2015
- Ziel: `sellingprice`
- Bereinigte Zeilen: 529.169
- Split: zufällig 80/20
- `random_state=42`

Architektur:

```text
VotingRegressor
├── XGBoost auf Rohpreis
└── XGBoost auf log1p(Preis)

Gewichtung: 50 % / 50 %
```

Zentrale Einstellungen:

```text
n_estimators=700
learning_rate=0.045
max_depth=9
min_child_weight=12
subsample=0.90
colsample_bytree=0.90
reg_alpha=0.02
reg_lambda=1.0
tree_method="hist"
```

Reproduktion:

```powershell
uv run python scripts/train_stage1_v2.py --max-rows 0
```

Der Lauf überschreibt das V2-Modell und seine Ergebnisdateien. Vorher Arbeitsstand sichern.

## 7. Wissenschaftlich strenger gemeinsamer Vergleich

V1 und V2 wurden mit `scripts/compare_stage1_v1_v2_shared_split.py` von Grund auf
neu trainiert. Beide erhielten exakt dieselben 423.335 Trainingszeilen und wurden
auf denselben 105.834 zuvor unangetasteten Testzeilen geprüft. Gespeicherte Modelle
wurden für diesen Vergleich weder geladen noch überschrieben.

Der strenge Vergleich bestätigt eine MAE-Verbesserung von **460,80 $ beziehungsweise
25,17 %**. Ein gepaartes Bootstrap-Verfahren mit 1.000 Wiederholungen ergibt ein
95%-Intervall von **450,74 $ bis 470,83 $**. Damit ist der Vorteil auf diesem Split
deutlich und nicht nur eine zufällige Schwankung.

Ergebnisdateien:

- `models/stage1_v1_v2_shared_split.json`
- `docs/stage1/model_results_stage1_v1_v2_shared_split.md`

Verbleibende Einschränkung: Der Split ist zufällig und kein zeitlicher Zukunftstest.

## 8. Integration in die App

`scripts/stage1_runtime.py` lädt zuerst:

```text
models/price_model_v2.joblib
```

Fehlt V2, wird automatisch V1 geladen:

```text
models/price_model.joblib
```

V1 wurde nicht überschrieben.

Zusätzliche V2-Eingaben:

- Ausstattungsvariante
- Getriebe
- Bundesstaat/Region
- Außenfarbe
- Innenfarbe

Die Kategorien werden aus dem trainierten Encoder gelesen. Die Oberfläche zeigt deutsche Werte, intern bleiben die englischen Trainingswerte erhalten:

| Oberfläche | Modellwert |
|---|---|
| Schwarz | `black` |
| Automatik | `automatic` |
| Limousine | `sedan` |
| Kalifornien (CA) | `ca` |

Marken und Modellnamen sind Eigennamen und werden nur korrekt formatiert.

### Kilometerumrechnung

Nutzer geben Kilometer ein; das US-Modell erwartet Meilen:

```text
Meilen = Kilometer × 0,621371
80.000 km ≈ 49.710 Meilen
```

### Zustandsauswahl

```text
Sehr gut → Gut → Durchschnittlich → Reparaturbedürftig → Stark reparaturbedürftig
```

| Oberfläche | Modellwert |
|---|---:|
| Sehr gut | 5,0 |
| Gut | 4,0 |
| Durchschnittlich | 3,0 |
| Reparaturbedürftig | 2,0 |
| Stark reparaturbedürftig | 1,0 |

### Sichtbarer Modellvergleich

Die App zeigt V1-MAE, V2-MAE, 461 $ weniger Fehler und 25,17 % Verbesserung aus dem gemeinsamen Neutraining.

## 9. Stage 2 mit V2

```text
Stage-2-Preis = V2-Basiswert × CPI-Multiplikator
```

| Auswertung | MAE |
|---|---:|
| V2 ohne CPI | 1.370,16 $ |
| V2 mit historischem CPI | 1.376,22 $ |

Änderung: +6,06 $ beziehungsweise +0,44 %. Stage 2 soll primär alte Preise auf ein späteres Marktpreisniveau übertragen. Im Testzeitraum 2014–2015 liegt der CPI fast bei 1,0. Das ist kein Widerspruch zur 25,17-%-Verbesserung von V2 gegenüber V1.

## 10. Stage 3 mit V2

Neue Faktoren: `models/seasonality_factors_v2.csv`

| Auswertung | MAE |
|---|---:|
| Ohne Stage 3 | 1.353,15 $ |
| Mit Stage 3 | 1.339,84 $ |

Verbesserung: 13,31 $ beziehungsweise 0,98 %. Stage 3 ist bewusst nur eine Feinkorrektur. Die Saisonprüfung ist kein vollständig unabhängiger neuer Stage-1-Test.

## 11. Relevante Dateien

| Datei | Aufgabe |
|---|---|
| `app/streamlit_app.py` | deutsche Oberfläche und Gesamtpipeline |
| `scripts/stage1_runtime.py` | V2 laden, V1-Fallback, Eingaben bauen |
| `scripts/train_stage1_v2.py` | vollständiges V2-Training |
| `scripts/optimize_stage1_v2.py` | Architektur- und Parametersuche |
| `models/price_model_v2.joblib` | V2-Produktionsmodell |
| `models/price_model.joblib` | V1-Fallback |
| `models/price_model_v2_metrics.json` | V2-Metriken und V1-Vergleich |
| `docs/stage1/model_results_stage1_v2.md` | lesbare V2-Ergebnisse |
| `scripts/stage2_macro.py` | CPI-Anpassung |
| `scripts/evaluate_stage2.py` | Stage-2-Evaluation mit V2 |
| `models/stage2_evaluation.json` | Stage-2-Rohresultate |
| `scripts/stage3_seasonality.py` | Saisonlogik mit V2 |
| `scripts/evaluate_stage3.py` | Stage-3-Evaluation |
| `models/seasonality_factors_v2.csv` | produktive V2-Saisonfaktoren |

## 12. Abhängigkeiten

`requirements.txt` enthält:

```text
streamlit==1.57.0
pandas==3.0.1
numpy==2.4.3
scikit-learn==1.8.0
joblib==1.5.3
xgboost==3.2.0
```

XGBoost ist auch beim Laden des gespeicherten Modells erforderlich.

## 13. Durchgeführte Tests

- Syntaxprüfung der App und Stage-1/2/3-Skripte
- Laden des V2-Modells
- echte V2-Vorhersage
- vollständiger Pfad V2 → CPI → Saison
- Streamlit-Test ohne App-Ausnahmen
- Prüfung der V2-Eingabefelder
- Prüfung von 1.831 $, 1.370 $ und 25,17 % in der Oberfläche
- Prüfung der Kilometer-zu-Meilen-Umrechnung
- Prüfung deutscher Zustände, Farben, Getriebe, Regionen und Karosserieformen
- vollständige Stage-2- und Stage-3-Neuauswertung

## 14. Bekannte Grenzen

1. Daten überwiegend aus 2014–2015.
2. US-Auktionspreise sind nicht direkt deutsche Endkundenpreise.
3. Zufälliger Split statt zeitlichem Zukunftstest.
4. August bis November fehlen für Stage 3 und bleiben neutral.
5. CPI beschreibt den Gesamtmarkt, nicht jedes Modell einzeln.
6. Region und Farbe können die Übertragbarkeit auf andere Märkte begrenzen.
7. Eine Unsicherheitsspanne fehlt noch.
8. Für eine zeitliche Generalisierungsprüfung wäre zusätzlich ein Zukunfts-Holdout sinnvoll.

## 15. Empfohlene nächste Schritte

1. Reproduzierbaren gemeinsamen V1/V2-Neutrainingsvergleich erstellen.
2. Preisintervall in der App ergänzen.
3. Ausstattungen passend zu Marke und Modell filtern.
4. Vollständigen V1/V2-End-to-End-Vergleich erstellen.
5. Aktuellere Fahrzeugdaten beschaffen.
6. LLM-Erklärungsschicht ergänzen.
7. Erst nach Teamprüfung Pull Request von `mail_project_pascal` nach `main` erstellen.

## 16. Hinweise für andere KI-Assistenten

- Entwicklungsbasis ist `mail_project_pascal`, nicht `main`.
- `main` nicht ohne ausdrückliche Freigabe verändern.
- V2 bevorzugen, V1-Fallback erhalten.
- MMR, VIN und Verkäufer nicht ungeprüft aufnehmen.
- Interne englische Modellkategorien nicht verändern; nur die Anzeige übersetzen.
- Oberfläche nutzt Kilometer, Modell nutzt Meilen.
- Stage 1 darf keinen Verkaufsmonat enthalten, solange Stage 2 und 3 getrennt bleiben.
- Für V2 `models/seasonality_factors_v2.csv` verwenden.
- Bei Metriken immer Split und Trainingshistorie dokumentieren.
- Die lokale Datei `PRAESI_STAGE2_STAGE3_ERKLAERUNG_UND_RUECKFRAGEN.md` nicht ungefragt committen.

## 17. Kurzfassung

> V2 nutzt mehr Daten, mehr Fahrzeugmerkmale und ein Ensemble aus zwei XGBoost-Modellen. Im gemeinsamen Neutraining sinkt der MAE von 1.831 $ auf 1.370 $, also um 25,17 %. Die App verwendet V2 standardmäßig und V1 als Fallback. Stage 2 aktualisiert das Marktpreisniveau, Stage 3 ergänzt eine kleine Saisonkorrektur. Die Oberfläche ist deutsch, nimmt Kilometer entgegen und übersetzt Eingaben intern in die Kategorien des amerikanischen Trainingsdatensatzes.
