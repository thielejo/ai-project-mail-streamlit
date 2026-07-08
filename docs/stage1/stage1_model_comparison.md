# Stage 1: Strenger Modellvergleich auf gemeinsamem Split

## Versuchsaufbau

V1 und V2 wurden in diesem Lauf von Grund auf neu trainiert. Beide Modelle
erhielten exakt dieselben Trainingszeilen und wurden anschließend auf exakt
denselben, zuvor unangetasteten Testzeilen bewertet. Gespeicherte Modelle
wurden nicht geladen und nicht überschrieben.

- Bereinigte Zeilen: 529,169
- Trainingszeilen: 423,335
- Testzeilen: 105,834
- Split: 80/20 mit `random_state=42`
- Testsplit-Fingerabdruck: `23cb375ec013b689d45cfb717cc8073e31129aabd45dae8cc74e83a894f3ab20`

## Ergebnis

| Modell | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| V1 – neu trainierter HistGradientBoostingRegressor | $1,830.95 | $3,276.81 | 0.8818 | 16.45% |
| V2 – neu trainiertes XGBoost-Ensemble | $1,370.15 | $2,400.34 | 0.9366 | 15.13% |

- MAE-Verbesserung: **$460.80**
- Relative MAE-Verbesserung: **25.17%**
- 95%-Bootstrap-Intervall der MAE-Verbesserung: **$450.74 bis $470.83**
- V2 hat auf 60.27% der einzelnen Testzeilen den kleineren absoluten Fehler.

## Interpretation

Dieser Vergleich beseitigt die Unsicherheit über eine mögliche Überschneidung
zwischen dem früheren V1-Training und dem V2-Testset. Die ausgewiesene
Verbesserung ist deshalb der methodisch bevorzugte V1/V2-Wert.

## Einschränkungen

- Der Split ist zufällig und kein zeitlicher Zukunftstest.
- Beide Modelle nutzen nur Zeilen, auf denen alle für V2 benötigten Merkmale vorhanden sind.
- Die Daten stammen aus US-Auktionen und überwiegend aus 2014–2015.
- V1 und V2 unterscheiden sich gleichzeitig in Modellarchitektur und Merkmalen; der Test misst den Gesamteffekt des V2-Upgrades.

## Reproduktion

```powershell
uv run python archive/scripts/legacy/compare_stage1_legacy_vs_production_shared_split.py --max-rows 0
```
