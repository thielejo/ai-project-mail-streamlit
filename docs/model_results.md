# Model Results

## Aktueller Modellstand

Das aktuelle Stage-1-Zielmodell ist ein getuntes **CatBoost-Modell mit Hubraum**
als zusaetzlichem Fahrzeugmerkmal. Stage 1 schaetzt den fahrzeugbezogenen
Basispreis. Stage 2 und Stage 3 ergaenzen danach Marktpreisniveau und
Saisonalitaet.

## Stage 1: Fahrzeugbezogener Basispreis

| Metrik | Wert |
|---|---:|
| MAE | $1.042 |
| RMSE | $1.875 |
| R2 | 0,961 |
| MAPE | 11,8 % |

Wichtigste Punkte:

- Bestes Stage-1-Modell: CatBoost mit FIN-/VIN-basiertem Hubraum.
- Der groesste Leistungsgewinn kommt aus CatBoost, Hubraum und Log-Zielgroesse.
- Zusaetzliche abgeleitete Features und Monotonie-Constraints wurden getestet,
  aber nicht uebernommen, weil sie die Modellleistung verschlechtert haben.

Details:

- [Aktuelles Stage-1-Modell](stage1/stage1_current_model.md)
- [Stage-1-Tuning](stage1/stage1_tuning_results.md)
- [Stage-1-Modellvergleich](stage1/stage1_model_comparison.md)
- [Benchmark getesteter Modellfamilien](stage1/stage1_benchmark_models.md)

## Stage 2: Marktpreisniveau

Stage 2 passt den historischen Basispreis ueber einen CPI-basierten
Marktfaktor an das gewaehlt Bewertungsdatum an.

Details:

- [Stage-2-Ergebnisse](stage2/model_results_stage2.md)

## Stage 3: Saisonalitaet

Stage 3 ergaenzt eine vorsichtige saisonale Korrektur nach Karosserieform und
Monat. Die Korrektur ist bewusst begrenzt und wird nur dort interpretiert, wo
ausreichend Datenbasis vorhanden ist.

Details:

- [Stage-3-Ergebnisse](stage3/model_results_stage3.md)

## Archivierter Altstand

Die fruehere Root-Datei `model_results.md` beschrieb noch das alte
HistGradientBoosting-Basismodell. Sie liegt jetzt als historischer Stand im
Archiv:

- `archive/docs/stage1/model_results_legacy_histgb.md`
