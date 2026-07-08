# Stage 1 V2 + FIN — Ergebnisse (Hubraum-Erweiterung)

Erweiterung des V2-Modells um den aus der VIN dekodierten **Hubraum** (`displacement`).
Vollständige Voruntersuchung: `vin_fin_enrichment/` (`FIN_Test.md`, `INTEGRATION_V2_FIN.md`).

## Ergebnis (voller Datensatz, 529.169 Zeilen, identischer Split)

| Modell | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Schlankes Stage 1 (HistGB) | $1.824 | $3.253 | 0,885 | 16,3 % |
| V2 (XGBoost-Ensemble) | $1.365 | $2.390 | 0,937 | 15,0 % |
| **V2 + Hubraum** | **$1.201** | **$2.040** | **0,954** | 13,7 % |

→ Hubraum bringt **−12 %** MAE zusätzlich zu V2 (eigenständiges Signal, kein Overlap mit trim/Farbe).

## Methode

- Pipeline identisch zu `scripts/train_stage1_v2.py` (XGBoost VotingRegressor 50/50,
  `reg:squarederror` roh + `reg:absoluteerror` log, OneHot `min_frequency=20`, 700 Bäume).
- Einzige Änderung: `displacement` als numerisches Feature (per VIN aus dem NHTSA-Decode gejoint).
- Hubraum-Abdeckung 99 %; fehlende Werte mit Median gefüllt.

## Belege über mehrere Stichproben

| Test | Vergleich | Stichprobe | Effekt Hubraum |
|---|---|---|---|
| 7 | vs. schlankes Stage-1 (HistGB) | 534k voll | +13,1 % |
| 8 | vs. echtes V2-Ensemble (XGBoost) | 529k voll | +12,2 % |

Per-Feature-Ablation (Test 6): Hubraum allein = alle drei FIN-Felder zusammen;
Kraftstoff/Zylinder redundant.

## Reproduktion

```bash
# Voraussetzung: VIN-Decode-Cache (regenerierbar, ~3 h)
uv run python vin_fin_enrichment/build_full_vin_cache.py
# Training V2 + Hubraum
uv run python scripts/train_stage1_v2_fin.py --max-rows 0
```

Artefakte: `models/price_model_v2_fin.joblib`, `models/price_model_v2_fin_metrics.json`.

## Status

Trainiert und dokumentiert. Noch **nicht** in die Streamlit-App / Stage 2–3 eingebunden
(V2 bleibt vorerst Produktionsmodell). Offener Schritt: Integration in App + Paper.
