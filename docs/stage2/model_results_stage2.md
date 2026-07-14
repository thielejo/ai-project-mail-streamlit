# Stage 2 Evaluation: CPI Macro Adjustment

## Methode

Stage 2 multipliziert den Stage-1-Basispreis mit dem CPI-Multiplikator
des Zieldatums:

```
stage2_price = stage1_price × cpi_multiplier(target_month)
```

Der `cpi_multiplier` ist auf den 2015-Jahresdurchschnitt (= 1.000) normiert.
Quelle: CPI Used Cars & Trucks (CUSR0000SETA02, FRED).

## Architekturgetreuer Backtest (2014–2015 Testset)

Als Basis dient das **CatBoost-Produktionsmodell (getunt, inkl. Hubraum)** — dasselbe
Modell, das die App ausliefert. Es enthält bewusst weder Verkaufsmonat noch
Makrovariable und liefert damit einen zeitneutralen Fahrzeug-Basiswert. Stage 2
wendet anschließend den CPI des tatsächlichen historischen Verkaufsmonats an.
Der Zielmonat wird so nicht doppelt gezählt.

| Metrik | Referenz-Baseline | Mit Stage 2 | Δ |
|---|---:|---:|---:|
| MAE | $1,056.54 | $1,075.86 | +19.32 |
| RMSE | $1,892.83 | $1,907.20 | — |
| R² | 0.9606 | 0.9600 | -0.0006 |
| MAPE | 11.91% | 12.02% | — |

**CPI-Multiplikator im Testset (2014–2015):**
- Min: 0.9838 / Max: 1.0290 / Ø 0.9995

> Der kleine Unterschied (+19.32 USD MAE) zeigt, wie Stage 2
> den zeitneutralen CatBoost-Basiswert im historischen Zeitraum verändert. Die
> Faktoren liegen nahe bei 1,0, weil 2014–2015 den Referenzzeitraum bilden.

## Vorwärtsprojektion (Median Stage-1-Preis: $12,400)

| Monat | CPI-Multiplikator | Stage 1 | Stage 2 | Δ % |
|---|---:|---:|---:|---:|
| 2015-01 | 0.9838 | $12,400 | $12,199 | -1.6% |
| 2016-01 | 0.9822 | $12,400 | $12,179 | -1.8% |
| 2017-01 | 0.9423 | $12,400 | $11,685 | -5.8% |
| 2018-01 | 0.9409 | $12,400 | $11,668 | -5.9% |
| 2019-01 | 0.9657 | $12,400 | $11,975 | -3.4% |
| 2020-01 | 0.9468 | $12,400 | $11,740 | -5.3% |
| 2020-06 | 0.9110 | $12,400 | $11,297 | -8.9% |
| 2021-01 | 1.0414 | $12,400 | $12,914 | +4.1% |
| 2021-06 | 1.3231 | $12,400 | $16,407 | +32.3% |
| 2021-12 | 1.4378 | $12,400 | $17,829 | +43.8% |
| 2022-01 | 1.4633 | $12,400 | $18,146 | +46.3% |
| 2022-06 | 1.4171 | $12,400 | $17,573 | +41.7% |
| 2022-12 | 1.3106 | $12,400 | $16,253 | +31.1% |
| 2023-01 | 1.2931 | $12,400 | $16,035 | +29.3% |
| 2023-09 | 1.2743 | $12,400 | $15,801 | +27.4% |
| 2024-01 | 1.2440 | $12,400 | $15,426 | +24.4% |
| 2025-01 | 1.2555 | $12,400 | $15,569 | +25.6% |
| 2026-01 | 1.2309 | $12,400 | $15,264 | +23.1% |
| 2026-06 | 1.2224 | $12,400 | $15,158 | +22.2% |

> Der COVID-bedingte Engpass (2021–2022) zeigt einen Preisanstieg von bis zu +22%.
> Der aktuelle Stand (2026-06) liegt stabil bei ~+22% über dem 2015-Niveau.

## Makro-Kontext 2026-06

| Indikator | Wert |
|---|---:|
| CPI-Multiplikator | 1.2224 |
| CPI Gebrauchtwagen (FRED) | 180.005 |
| Leitzins Fed Funds % | 3.63 |
| Konsumentenstimmung (Univ. Michigan) | 44.8 |
| Arbeitslosenquote % | 4.2 |
| Fahrzeugverkäufe Mio. SAAR | 16.949 |
| Rezession NBER (0/1) | 0.0 |
| High-Yield-Spread % | 2.75 |

## Einschränkungen

- Stage 2 extrapoliert ausschließlich über CPI-Inflation; strukturelle Marktveränderungen
  (z. B. Elektrifizierung, Chip-Engpässe) sind nicht modelliert.
- Für Monate ohne FRED-Daten wird der letzte verfügbare Monat genutzt (Forward-Fill).
- Im gespeicherten Makrostand enthält 2026-05 mit 180.005 eine gegenüber
  2026-04 veränderte CPI-Beobachtung. Die Werte für 2026-06 und 2026-07
  entsprechen dagegen dem Maiwert und sind daher vorläufige Fortschreibungen,
  keine neuen unabhängigen CPI-Messpunkte.
- Das CatBoost-Produktionsmodell enthält bewusst weder `sale_month` noch
  `year_month`. Dadurch bleiben Marktbewegung und Saison vollständig in
  Stage 2 und Stage 3.
