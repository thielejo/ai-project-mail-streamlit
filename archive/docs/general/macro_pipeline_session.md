# Entwicklungssitzung — 04. Juni 2026

## Überblick

In dieser Sitzung wurden die Grundlagen für das 3-stufige Ensemble-Modell gelegt:
die Datenpipeline wurde aufgebaut, Makrodaten integriert und Stage 1 (XGBoost Micro-Modell) trainiert.

---

## 1. Repository-Synchronisation

**Ausgangslage:** Der Branch `Mail_project_moritz` war nicht auf dem Stand von `main`.
Tara hatte zwischenzeitlich auf `main` folgende Dateien gepusht:

| Datei | Inhalt |
|---|---|
| `car_prices_clean.csv` | Bereinigter Manheim-Datensatz (558.743 Zeilen) |
| `scripts/clean_car_prices.py` | Reproduzierbares Cleaning-Skript |
| `data_cleaning.md` | Dokumentation der Cleaning-Schritte |
| `01_exploration.ipynb` | Exploratives Analyse-Notebook |

**Aktion:** `git merge origin/main` in `Mail_project_moritz`.

---

## 2. Bereinigter Datensatz (Taras Arbeit)

Datei: `car_prices_clean.csv`

- **Quelle:** Manheim Used Car Auction Dataset (`tunguz/used-car-auction-prices` auf Kaggle)
- **Zeilen:** 558.743
- **Zeitraum:** 2014–2015 (US-Auktionsmarkt)
- **Spalten:** `year, make, model, trim, body, transmission, vin, state, condition, odometer, color, interior, seller, mmr, sellingprice, saledate`

Cleaning-Schritte (Details in `data_cleaning.md`):
- 26 fehlerhafte CSV-Zeilen repariert (Komma im `trim`-Wert)
- 94 Zeilen ohne `sellingprice` oder `odometer` entfernt
- Kategorische Spalten normalisiert (lowercase, Whitespace, Tippfehler)
- `saledate` in UTC-Datetime konvertiert

---

## 3. Makrodaten-Pipeline (`scripts/enrich_macro.py`)

### Warum Makrodaten?

Das XGBoost-Modell (Stage 1) wird auf 2014-2015-Daten trainiert und liefert Preise in
2015-Dollar. Um aktuelle Preise zu schätzen, wird der Stage-1-Output mit einem
Marktmultiplikator aus Makrodaten skaliert (Stage 2).

### FRED-Serien

Alle Daten werden kostenlos und ohne API-Key direkt von der Federal Reserve
Economic Data (FRED) geladen:

| Spaltenname | FRED-Serie | Inhalt | Verwendungszweck |
|---|---|---|---|
| `cpi_used_cars` | `CUSR0000SETA01` | CPI Used Cars & Trucks | MUVVI-Proxy, Kern des Preismultiplikators |
| `fedfunds` | `FEDFUNDS` | US Federal Funds Rate | Kreditkosten, Nachfragedämpfer |
| `consumer_sentiment` | `UMCSENT` | Univ. of Michigan Sentiment | Kaufbereitschaft der Konsumenten |
| `unemployment` | `UNRATE` | US Arbeitslosenquote | Kaufkraft der Haushalte |
| `total_vehicle_sales` | `TOTALSA` | Gesamtfahrzeugverkäufe (SAAR) | Marktvolumen |
| `oil_price_wti` | `DCOILWTICO` | WTI Rohölpreis (USD/Barrel) | Nachfrage nach SUVs vs. Sedans |
| `recession` | `USREC` | NBER Rezessionsindikator (0/1) | Black-Swan-Flag für Krisenperioden |
| `credit_spread` | `BAMLH0A0HYM2` | High-Yield Credit Spread | Marktpanik / Kreditstress |

### Der `cpi_multiplier`

Kernformel für Stage 2:

```
Live Price = Stage1_Baseline × cpi_multiplier × seasonal_factor
```

Der `cpi_multiplier` normiert den CPI auf das Basisjahr 2015 (Wert = 1.0).
Beispiel: Ein Wert von 1.20 im Jahr 2024 bedeutet, dass Gebrauchtwagen im
Durchschnitt 20% teurer sind als 2015.

### Outputs

| Datei | Inhalt |
|---|---|
| `macro_index.csv` | Vollständige FRED-Historie (1950er bis heute) für Stage 2 |
| `car_prices_macro.csv` | Fahrzeugdatensatz + Makrowerte per Verkaufsmonat (gitignored, 98 MB) |

### Warum nicht der offizielle MUVVI?

Der Manheim Used Vehicle Value Index (MUVVI) wird von Cox Automotive per
hedonischer Regression berechnet und ist proprietär (kostenpflichtig).
`CUSR0000SETA01` ist der beste frei verfügbare Proxy, da das Bureau of Labor
Statistics Auktionspreise als Datenquelle einbezieht. Scraping der Manheim-Website
ist laut deren ToS untersagt.

---

## 4. Stage 1: XGBoost Micro-Modell (`scripts/train_stage1.py`)

### Features

Bewusst **ohne** `mmr` (Manheim Market Report) trainiert, damit das Modell
echte Fahrzeugeigenschaften lernt statt nur den Händlerschätzwert nachzuahmen.

| Feature | Typ | Beschreibung |
|---|---|---|
| `make` | kategorial | Fahrzeugmarke |
| `body` | kategorial | Karosserieform (SUV, Sedan, ...) |
| `transmission` | kategorial | Getriebe |
| `condition` | numerisch | Manheim Condition Score (1–5) |
| `odometer` | numerisch | Kilometerstand |
| `vehicle_age` | numerisch | Alter zum Verkaufszeitpunkt (sale_year − year) |

Outlier-Filter (bewusst behalten in `car_prices_clean.csv`, hier entfernt):
- `sellingprice < 500` oder `> 150.000 USD`
- `odometer > 500.000`

### Ergebnisse

| Metrik | Wert |
|---|---|
| **MAE** | **$2.451 (17.6%)** |
| **RMSE** | $3.951 |
| Durchschnittlicher Verkaufspreis (Test) | $13.947 |
| Trainingszeilen | 423.910 |
| Testzeilen | 105.978 |
| Bäume (n_estimators) | 2.000 |

**Feature Importance:**

| Feature | Wichtigkeit |
|---|---|
| `odometer` | 27% |
| `vehicle_age` | 25% |
| `make` | 20% |
| `body` | 17% |
| `condition` | 9% |
| `transmission` | 1% |

### Einordnung

- 17.6% MAE ist solide für ein reines Fahrzeugattribut-Modell ohne `mmr`
- Mit `mmr` wären < 5% erreichbar — aber das wäre methodisch unehrlich
- Das Modell lernt plausible Zusammenhänge: Kilometerstand und Alter dominieren
- Weitere Optimierung durch zusätzliche Features möglich (z.B. `state`)

### Gespeicherte Artefakte

| Datei | Inhalt |
|---|---|
| `models/stage1_xgboost.json` | Trainiertes XGBoost-Modell |
| `models/stage1_encoder.pkl` | OrdinalEncoder für kategorische Features |

---

## 5. Nächste Schritte

- [ ] Stage 2: `cpi_multiplier` aus `macro_index.csv` auf Stage-1-Output anwenden
- [ ] Stage 3: Saisonale Faktoren nach `body`-Typ und Verkaufsmonat
- [ ] Streamlit-App: Fahrzeugrechner mit Toggle für Marktanpassung
- [ ] Weitere Features evaluieren (z.B. `state` für regionale Preisunterschiede)
