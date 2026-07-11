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

Stage 1 V2 enthält bewusst weder Verkaufsmonat noch Makrovariable und liefert
damit einen zeitneutralen Fahrzeug-Basiswert. Stage 2 wendet anschließend den
CPI des tatsächlichen historischen Verkaufsmonats an. Der Zielmonat wird so
nicht doppelt gezählt.

| Metrik | Referenz-Baseline | Mit Stage 2 | Δ |
|---|---:|---:|---:|
| MAE | $1,370.16 | $1,383.73 | +13.57 |
| RMSE | $2,400.34 | $2,407.08 | — |
| R² | 0.9366 | 0.9362 | -0.0004 |
| MAPE | 15.13% | 15.21% | — |

**CPI-Multiplikator im Testset (2014–2015):**
- Min: 0.9838 / Max: 1.0290 / Ø 0.9995

> Der kleine Unterschied (+13.57 USD MAE) zeigt, wie Stage 2
> den zeitneutralen V2-Basiswert im historischen Zeitraum verändert. Die
> Faktoren liegen nahe bei 1,0, weil 2014–2015 den Referenzzeitraum bilden.

## Vorwärtsprojektion (Median Stage-1-Preis: $12,495)

| Monat | CPI-Multiplikator | Stage 1 | Stage 2 | Δ % |
|---|---:|---:|---:|---:|
| 2015-01 | 0.9838 | $12,495 | $12,293 | -1.6% |
| 2016-01 | 0.9822 | $12,495 | $12,273 | -1.8% |
| 2017-01 | 0.9423 | $12,495 | $11,774 | -5.8% |
| 2018-01 | 0.9409 | $12,495 | $11,757 | -5.9% |
| 2019-01 | 0.9657 | $12,495 | $12,067 | -3.4% |
| 2020-01 | 0.9468 | $12,495 | $11,830 | -5.3% |
| 2020-06 | 0.9110 | $12,495 | $11,383 | -8.9% |
| 2021-01 | 1.0414 | $12,495 | $13,013 | +4.1% |
| 2021-06 | 1.3231 | $12,495 | $16,533 | +32.3% |
| 2021-12 | 1.4378 | $12,495 | $17,965 | +43.8% |
| 2022-01 | 1.4633 | $12,495 | $18,285 | +46.3% |
| 2022-06 | 1.4171 | $12,495 | $17,707 | +41.7% |
| 2022-12 | 1.3106 | $12,495 | $16,377 | +31.1% |
| 2023-01 | 1.2931 | $12,495 | $16,158 | +29.3% |
| 2023-09 | 1.2743 | $12,495 | $15,922 | +27.4% |
| 2024-01 | 1.2440 | $12,495 | $15,544 | +24.4% |
| 2025-01 | 1.2555 | $12,495 | $15,688 | +25.6% |
| 2026-01 | 1.2309 | $12,495 | $15,381 | +23.1% |
| 2026-06 | 1.2224 | $12,495 | $15,274 | +22.2% |

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
- Im gespeicherten Makrostand sind die CPI-Werte für 2026-05 und 2026-06 aus
  2026-04 fortgeschrieben. 2026-06 ist daher ein Bewertungsdatum, kein neuer
  unabhängiger CPI-Messpunkt.
- Stage 1 V2 enthält bewusst weder `sale_month` noch `year_month`. Dadurch
  bleiben Marktbewegung und Saison vollständig in Stage 2 und Stage 3.
