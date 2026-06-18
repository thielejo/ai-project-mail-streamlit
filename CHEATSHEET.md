# Project Cheat Sheet — Team MAIL

Schnellreferenz für Präsentationen und Prof-Fragen.

---

## Das Projekt in einem Satz

Wir bauen einen KI-Agenten, der den aktuellen Marktwert eines Gebrauchtwagens berechnet — indem er fahrzeugspezifische Merkmale (XGBoost), makroökonomische Lage (CPI/MUVVI, Zinsen) und Saisonalität nach Karosserietyp kombiniert.

---

## Die Kernformel

```
Live-Preis = Basispreis (Stage 1) × Marktmultiplikator (Stage 2) × Saisonfaktor (Stage 3)
```

---

## Stage 1 — XGBoost Micro-Modell

**Was macht es?**
Lernt den "physischen Wert" eines Fahrzeugs aus seinen Attributen. Output ist ein Basispreis in 2015-Dollar (trainiert auf 2014–2015 Auktionsdaten).

**Features:**
| Feature | Wichtigkeit |
|---|---|
| `odometer` | 27 % |
| `vehicle_age` | 25 % |
| `make` | 20 % |
| `body` | 17 % |
| `condition` | 9 % |
| `transmission` | 1 % |

**Ergebnisse:**
- MAE: **$2.451** (17,6 % des Durchschnittspreises)
- RMSE: $3.951
- Ø Verkaufspreis im Testset: $13.947
- Trainingszeilen: 423.910 | Testzeilen: 105.978
- Bäume: 2.000

**Warum kein MMR-Feature?**
MMR (Manheim Market Report) ist selbst eine fahrzeugspezifische Preisschätzung — das wäre fast Data Leakage. Mit MMR wären < 5 % MAE möglich, aber das Modell würde nichts "lernen", sondern nur den Händlerschätzwert nachahmen. Wir wollen, dass das Modell echte Fahrzeugeigenschaften versteht.

**Warum kein `state`-Feature?**
Möglich und würde regionale Preisunterschiede abbilden. Bewusst erstmal weggelassen, um das Modell einfach zu halten. Ist als optionaler nächster Schritt notiert.

---

## Stage 2 — Makro-Modell (Expertensystem)

**Was macht es?**
Skaliert den Basispreis aus Stage 1 auf heutige Marktbedingungen. Kernmechanismus: `cpi_multiplier` normiert den CPI-Index auf das Basisjahr 2015.

**Warum kein klassisches ML für Stage 2?**
"Strukturelle Brüche" — Black-Swan-Ereignisse wie die Pandemie (2020–2022) führten dazu, dass steigende Zinsen mit *steigenden* Autopreisen korrelierten (umgekehrt zu normalem Verhalten wegen Chipmangel). ML-Modelle lernen solche Ausnahmesituationen als Regel und generalisieren falsch. Ein deterministisches Expertensystem ist robuster.

**FRED-Serien in `macro_index.csv`:**
| Spalte | FRED-Serie | Bedeutung |
|---|---|---|
| `cpi_used_cars` | CUSR0000SETA01 | CPI für Gebrauchtwagen (MUVVI-Proxy) |
| `fedfunds` | FEDFUNDS | US-Leitzins |
| `consumer_sentiment` | UMCSENT | Verbraucherstimmung (UoM) |
| `unemployment` | UNRATE | US-Arbeitslosenquote |
| `total_vehicle_sales` | TOTALSA | Gesamtfahrzeugverkäufe (SAAR) |
| `recession` | USREC | NBER Rezessionsindikator (0/1) |
| `credit_spread` | BAMLH0A0HYM2 | High-Yield-Risikoaufschlag |
| `cpi_multiplier` | — | Berechneter Multiplikator (Basis 2015 = 1.0) |

**Beispiel cpi_multiplier:** Wert 1.20 im Jahr 2024 → Gebrauchtwagen sind im Schnitt 20 % teurer als 2015.

**Warum nicht den echten MUVVI?**
MUVVI von Cox Automotive/Manheim ist proprietär und kostenpflichtig. `CUSR0000SETA01` (BLS) ist der beste frei verfügbare Proxy, weil das Bureau of Labor Statistics Auktionspreise als Datenquelle einbezieht. Scraping der Manheim-Website verstößt gegen deren ToS.

---

## Stage 3 — Saisonalitäts-Modell

**Was macht es?**
Passt den Preis nach Karosserietyp und Verkaufsmonat an.

**Logik:**
- **Cabrios & Sportwagen:** Nachfragepeak im Frühsommer (Juni/Juli), Einbruch im Winter
- **SUVs & 4×4:** Sicherheitsbedürfnis vor Winter, Peak Oktober/November
- **Budget-Sedans & Kompakte:** "Spring Bounce" Feb–Apr (US-Steuerrückerstattungen)

*(Stage 3 ist noch nicht implementiert — folgt nach Stage 2)*

---

## Datensatz

**Quelle:** Manheim Used Car Auction Dataset (Kaggle: `tunguz/used-car-auction-prices`)

- 558.743 Zeilen nach Cleaning
- Zeitraum: 2014–2015, US-Markt
- Wholesale B2B-Auktionspreise (kein Händleraufschlag)
- Cleaning durch Tara: 26 fehlerhafte Zeilen repariert, 94 Outlier entfernt, Datumsformat normalisiert

**Warum nur USA?**
Einheitliche Währung (USD), konsistente Marktstruktur, kein EUR/GBP-Mischmasch, FRED-Daten passen direkt zum Datensatz.

**Warum 2014–2015?**
Das sind die Transaktionsdaten im Datensatz. Stage 2 und 3 bringen den Output in die Gegenwart.

---

## Endprodukt

**Streamlit-App** mit:
- Fahrzeugformular (Marke, Modell, Kilometerstand, Zustand etc.)
- Preisausgabe: Basispreis → Marktbereinigt → Saisonal angepasst
- **LLM-Orchestrator** (Ollama/lokale API): wandelt die Zahlen in natürlichsprachliche Handlungsempfehlung um

Beispiel-Output des LLM:
> *"Angesichts des hohen Federal Funds Rate und der Tatsache, dass Ihr Fahrzeug ein Cabrio ist, empfehlen wir, im November nicht zu verkaufen. Halten Sie das Fahrzeug 5 Monate, um den Spring Bounce zu nutzen — das maximiert Ihre Rendite um geschätzte $1.800."*

---

## Häufige Prof-Fragen

**"Warum XGBoost und nicht ein neuronales Netz?"**
XGBoost ist bei tabellarischen Daten mit gemischten Feature-Typen (kategorial + numerisch) in der Regel konkurrenzfähig oder besser als NNs — und deutlich interpretierbarer. Feature Importance ist bei XGBoost direkt ableitbar. Für unser Ensemble-Design (3 getrennte Stufen) ist Interpretierbarkeit zentral.

**"Ist euer Modell nicht zu simpel?"**
Die Stärke liegt im Ensemble-Design, nicht in einem einzelnen Modell. Stage 1 allein ist simpel — aber Stage 1 + 2 + 3 + LLM ergibt ein System, das kein einzelnes Modell replizieren könnte, weil es Wissen aus drei verschiedenen Domänen kombiniert.

**"Warum habt ihr MMR nicht genutzt? Das wäre doch genauer."**
Genauer, ja — aber methodisch fragwürdig. MMR ist eine Schätzung des Zielwerts für exakt dieses Fahrzeug. Das Modell würde lernen: "MMR + Rauschen = Preis". Das ist kein ML, das ist Lookup. Unser Modell ohne MMR zeigt, was die Merkmale des Fahrzeugs wirklich erklären.

**"Was ist der Unterschied zwischen MMR und MUVVI?"**
- **MMR** = fahrzeugspezifisch (dieser Ford Focus ist $11.400 wert) → zu nah am Zielwert
- **MUVVI** = aggregierter Marktindex (Gebrauchtwagen insgesamt sind 3 % teurer als letztes Jahr) → legitimer Marktkontext, kein Fahrzeugbezug

**"Warum kein Ende-zu-Ende-Modell statt 3 Stufen?"**
Wegen der strukturellen Brüche (Pandemie, Chipmangel). Ein Ende-zu-Ende-Modell würde diese Schocks in die Fahrzeugattribute "einbacken" und bei normalem Markt falsch liegen. Die Trennung ermöglicht, jeden Faktor separat zu erklären und zu aktualisieren.

**"Wie aktuell sind eure Makrodaten?"**
`macro_index.csv` wird via `enrich_macro.py` direkt von FRED geladen — immer bis zum aktuellen Monat. Stand der letzten Generierung: Juni 2026.

**"Welchen Beitrag hat wer geleistet?"**
- **Tara:** Datenbeschaffung, Cleaning-Pipeline (`clean_car_prices.py`), Explorations-Notebook
- **Moritz:** Makrodaten-Pipeline (`enrich_macro.py`), Stage 1 XGBoost-Training, Präsentation
- **Johanna & Pasca:** *(hier euren jeweiligen Beitrag ergänzen)*

---

## Wichtige Dateien

| Datei | Inhalt |
|---|---|
| `car_prices_clean.csv` | Bereinigter Manheim-Datensatz (558.743 Zeilen) |
| `macro_index.csv` | FRED-Makrodaten 1996–2026 (9 Spalten) |
| `car_prices_macro.csv` | Fahrzeugdaten + Makrowerte (gitignored, 98 MB) |
| `models/stage1_xgboost.json` | Trainiertes XGBoost-Modell |
| `models/stage1_encoder.pkl` | OrdinalEncoder für kategorische Features |
| `scripts/enrich_macro.py` | FRED-Download + macro_index.csv generieren |
| `scripts/train_stage1.py` | Stage 1 Training |
| `project_proposal_v2.md` | Vollständiges Projektkonzept |

---

## Nächste Schritte (Stand Juni 2026)

- [ ] Stage 2: `cpi_multiplier` auf Stage-1-Output anwenden
- [ ] Stage 3: Saisonale Faktoren nach Body-Typ
- [ ] Streamlit-App: Fahrzeugrechner
- [ ] LLM-Orchestrator: natürlichsprachliche Empfehlung
- [ ] MUVVI vs. CPI: Rücksprache mit Prof
- [ ] Optional: `state` als regionales Feature
