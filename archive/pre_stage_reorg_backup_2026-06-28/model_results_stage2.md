# Stage 2 Evaluation: CPI Macro Adjustment

## Methode

Stage 2 multipliziert den Stage-1-Basispreis mit dem CPI-Multiplikator
des Zieldatums:

```
stage2_price = stage1_price × cpi_multiplier(target_month)
```

Der `cpi_multiplier` ist auf den 2015-Jahresdurchschnitt (= 1.000) normiert.
Quelle: CPI Used Cars & Trucks (CUSR0000SETA01, FRED).

## Architekturgetreuer Backtest (2014–2015 Testset)

Stage 1 V2 enthält bewusst weder Verkaufsmonat noch Makrovariable und liefert
damit einen zeitneutralen Fahrzeug-Basiswert. Stage 2 wendet anschließend den
CPI des tatsächlichen historischen Verkaufsmonats an. Der Zielmonat wird so
nicht doppelt gezählt.

| Metrik | Referenz-Baseline | Mit Stage 2 | Δ |
|---|---:|---:|---:|
| MAE | $1,370.16 | $1,376.22 | +6.06 |
| RMSE | $2,400.34 | $2,411.11 | — |
| R² | 0.9366 | 0.9360 | -0.0006 |
| MAPE | 15.13% | 15.11% | — |

**CPI-Multiplikator im Testset (2014–2015):**
- Min: 0.9897 / Max: 1.0024 / Ø 0.9967

> Der kleine Unterschied (+6.06 USD MAE) zeigt, wie Stage 2
> den zeitneutralen V2-Basiswert im historischen Zeitraum verändert. Die
> Faktoren liegen nahe bei 1,0, weil 2014–2015 den Referenzzeitraum bilden.

## Vorwärtsprojektion (Median Stage-1-Preis: $12,495)

| Monat | CPI-Multiplikator | Stage 1 | Stage 2 | Δ % |
|---|---:|---:|---:|---:|
| 2015-01 | 0.9930 | $12,495 | $12,408 | -0.7% |
| 2016-01 | 0.9971 | $12,495 | $12,459 | -0.3% |
| 2017-01 | 1.0070 | $12,495 | $12,583 | +0.7% |
| 2018-01 | 1.0044 | $12,495 | $12,551 | +0.4% |
| 2019-01 | 1.0045 | $12,495 | $12,551 | +0.4% |
| 2020-01 | 1.0050 | $12,495 | $12,558 | +0.5% |
| 2020-06 | 0.9953 | $12,495 | $12,437 | -0.5% |
| 2021-01 | 1.0191 | $12,495 | $12,734 | +1.9% |
| 2021-06 | 1.0482 | $12,495 | $13,097 | +4.8% |
| 2021-12 | 1.1396 | $12,495 | $14,240 | +14.0% |
| 2022-01 | 1.1430 | $12,495 | $14,282 | +14.3% |
| 2022-06 | 1.1681 | $12,495 | $14,596 | +16.8% |
| 2022-12 | 1.2064 | $12,495 | $15,074 | +20.6% |
| 2023-01 | 1.2088 | $12,495 | $15,105 | +20.9% |
| 2023-09 | 1.2200 | $12,495 | $15,244 | +22.0% |
| 2024-01 | 1.2176 | $12,495 | $15,214 | +21.8% |
| 2025-01 | 1.2134 | $12,495 | $15,162 | +21.3% |
| 2026-01 | 1.2179 | $12,495 | $15,218 | +21.8% |
| 2026-06 | 1.2177 | $12,495 | $15,216 | +21.8% |

> Der COVID-bedingte Engpass (2021–2022) zeigt einen Preisanstieg von bis zu +22%.
> Der aktuelle Stand (2026-06) liegt stabil bei ~+22% über dem 2015-Niveau.

## Makro-Kontext 2026-06

| Indikator | Wert |
|---|---:|
| CPI-Multiplikator | 1.2177 |
| CPI Gebrauchtwagen (FRED) | 179.174 |
| Leitzins Fed Funds % | 3.63 |
| Konsumentenstimmung (Univ. Michigan) | 49.8 |
| Arbeitslosenquote % | 4.3 |
| Fahrzeugverkäufe Mio. SAAR | 16.485 |
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
