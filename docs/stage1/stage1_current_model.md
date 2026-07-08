# Stage 1 — Aktuelles Modell

## Kurzfassung

Das aktuelle Stage-1-Zielmodell ist ein getuntes **CatBoost-Modell mit Hubraum**
als zusätzlichem Fahrzeugmerkmal. Stage 1 berechnet den fahrzeugbezogenen
Basispreis. Markt- und Saisonanpassungen bleiben davon getrennt und werden in
Stage 2 und Stage 3 ergänzt.

## Modellstand

| Bestandteil | Aktueller Stand |
|---|---|
| Modelltyp | CatBoost |
| Wichtigste Erweiterung | Hubraum aus FIN/VIN-Dekodierung |
| Zieltransformation | Log-Ziel |
| Loss | MAE |
| Trainingsbasis | voller bereinigter Datensatz |
| Status | finaler Stage-1-Kandidat / Zielmodell |

## Ergebnis

| Metrik | Wert |
|---|---:|
| MAE | $1.042 |
| RMSE | $1.875 |
| R² | 0,961 |
| MAPE | 11,8 % |

Damit ist das getunte CatBoost-Modell der bisher stärkste Stage-1-Ansatz. Der
größte Leistungsgewinn kommt aus der Kombination aus CatBoost, Hubraum und
Log-Zielgröße. Zusätzliche abgeleitete Features und Monotonie-Constraints wurden
getestet, aber verworfen, weil sie die Modellleistung verschlechtert haben.

## Einordnung

Der verbleibende Fehler liegt vor allem an Auktionsrauschen und fehlenden
Fahrzeuginformationen wie Unfallhistorie, Ausstattung oder Servicehistorie. Diese
Grenzen werden in der App über eine Datenbasis- und Vertrauensbewertung
transparent gemacht.

Details zum Tuning stehen in
[`stage1_tuning_results.md`](stage1_tuning_results.md). Der neutrale
Modellvergleich steht in
[`stage1_model_comparison.md`](stage1_model_comparison.md).
