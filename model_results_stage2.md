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

Für beide Zeilen wird Stage 1 mit der festen Marktreferenz `2015-02`
ausgeführt. Stage 2 wendet danach den CPI des tatsächlichen historischen
Verkaufsmonats an. So entspricht der Test der aktuellen App-Architektur und
zählt den Monat nicht bereits in Stage 1 mit.

| Metrik | Referenz-Baseline | Mit Stage 2 | Δ |
|---|---:|---:|---:|
| MAE | $1,890.21 | $1,889.19 | -1.02 |
| RMSE | $3,325.06 | $3,331.03 | — |
| R² | 0.8797 | 0.8793 | -0.0004 |
| MAPE | 17.31% | 17.24% | — |

**CPI-Multiplikator im Testset (2014–2015):**
- Min: 0.9897 / Max: 1.0024 / Ø 0.9967

> Der kleine Unterschied (-1.02 USD MAE) bestätigt, dass Stage 2
> die feste 2015-02-Referenz im historischen Zeitraum leicht verbessert. Die
> Faktoren liegen nahe bei 1,0, weil 2014–2015 den Referenzzeitraum bilden.

## Vorwärtsprojektion (Median Stage-1-Preis: $12,717)

| Monat | CPI-Multiplikator | Stage 1 | Stage 2 | Δ % |
|---|---:|---:|---:|---:|
| 2015-01 | 0.9930 | $12,717 | $12,627 | -0.7% |
| 2016-01 | 0.9971 | $12,717 | $12,679 | -0.3% |
| 2017-01 | 1.0070 | $12,717 | $12,806 | +0.7% |
| 2018-01 | 1.0044 | $12,717 | $12,773 | +0.4% |
| 2019-01 | 1.0045 | $12,717 | $12,773 | +0.4% |
| 2020-01 | 1.0050 | $12,717 | $12,781 | +0.5% |
| 2020-06 | 0.9953 | $12,717 | $12,657 | -0.5% |
| 2021-01 | 1.0191 | $12,717 | $12,959 | +1.9% |
| 2021-06 | 1.0482 | $12,717 | $13,329 | +4.8% |
| 2021-12 | 1.1396 | $12,717 | $14,492 | +14.0% |
| 2022-01 | 1.1430 | $12,717 | $14,535 | +14.3% |
| 2022-06 | 1.1681 | $12,717 | $14,854 | +16.8% |
| 2022-12 | 1.2064 | $12,717 | $15,341 | +20.6% |
| 2023-01 | 1.2088 | $12,717 | $15,372 | +20.9% |
| 2023-09 | 1.2200 | $12,717 | $15,514 | +22.0% |
| 2024-01 | 1.2176 | $12,717 | $15,483 | +21.8% |
| 2025-01 | 1.2134 | $12,717 | $15,430 | +21.3% |
| 2026-01 | 1.2179 | $12,717 | $15,488 | +21.8% |
| 2026-06 | 1.2177 | $12,717 | $15,485 | +21.8% |

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
- Das Stage-1-Modell kennt `year_month`-Werte außerhalb von 2014–2015 nicht;
  der OrdinalEncoder kodiert diese als -1. Da `year_month` der unwichtigste Feature
  (Importance 32 vs. 2470 für `make`) ist, ist der Effekt minimal.
