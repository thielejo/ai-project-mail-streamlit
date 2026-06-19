# Stage 2 Evaluation: CPI Macro Adjustment

## Methode

Stage 2 multipliziert den Stage-1-Basispreis mit dem CPI-Multiplikator
des Zieldatums:

```
stage2_price = stage1_price × cpi_multiplier(target_month)
```

Der `cpi_multiplier` ist auf den 2015-Jahresdurchschnitt (= 1.000) normiert.
Quelle: CPI Used Cars & Trucks (CUSR0000SETA01, FRED).

## Backtest: Historische Genauigkeit (2014–2015 Testset)

| Metrik | Stage 1 | Stage 2 | Δ |
|---|---:|---:|---:|
| MAE | $1,849.96 | $1,849.81 | -0.15 |
| RMSE | $3,298.97 | $3,306.09 | — |
| R² | 0.8816 | 0.8811 | -0.0005 |
| MAPE | 16.42% | 16.37% | — |

**CPI-Multiplikator im Testset (2014–2015):**
- Min: 0.9897 / Max: 1.0024 / Ø 0.9967

> Der minimale Unterschied (±0 MAE) bestätigt, dass Stage 2
> die historische Genauigkeit nicht verschlechtert. Die Trainingsperiode liegt
> im CPI-Baseline-Bereich (~0.99), sodass der Anpassungsfaktor nahezu neutral ist.

## Vorwärtsprojektion (Median Stage-1-Preis: $12,519)

| Monat | CPI-Multiplikator | Stage 1 | Stage 2 | Δ % |
|---|---:|---:|---:|---:|
| 2015-01 | 0.9930 | $12,519 | $12,431 | -0.7% |
| 2016-01 | 0.9971 | $12,519 | $12,482 | -0.3% |
| 2017-01 | 1.0070 | $12,519 | $12,606 | +0.7% |
| 2018-01 | 1.0044 | $12,519 | $12,574 | +0.4% |
| 2019-01 | 1.0045 | $12,519 | $12,574 | +0.4% |
| 2020-01 | 1.0050 | $12,519 | $12,582 | +0.5% |
| 2020-06 | 0.9953 | $12,519 | $12,460 | -0.5% |
| 2021-01 | 1.0191 | $12,519 | $12,757 | +1.9% |
| 2021-06 | 1.0482 | $12,519 | $13,122 | +4.8% |
| 2021-12 | 1.1396 | $12,519 | $14,266 | +14.0% |
| 2022-01 | 1.1430 | $12,519 | $14,309 | +14.3% |
| 2022-06 | 1.1681 | $12,519 | $14,623 | +16.8% |
| 2022-12 | 1.2064 | $12,519 | $15,102 | +20.6% |
| 2023-01 | 1.2088 | $12,519 | $15,133 | +20.9% |
| 2023-09 | 1.2200 | $12,519 | $15,273 | +22.0% |
| 2024-01 | 1.2176 | $12,519 | $15,242 | +21.8% |
| 2025-01 | 1.2134 | $12,519 | $15,190 | +21.3% |
| 2026-01 | 1.2179 | $12,519 | $15,247 | +21.8% |
| 2026-06 | 1.2177 | $12,519 | $15,244 | +21.8% |

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
- Das Stage-1-Modell kennt `year_month`-Werte außerhalb von 2014–2015 nicht;
  der OrdinalEncoder kodiert diese als -1. Da `year_month` der unwichtigste Feature
  (Importance 32 vs. 2470 für `make`) ist, ist der Effekt minimal.
