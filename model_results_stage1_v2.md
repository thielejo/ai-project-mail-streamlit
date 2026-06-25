# Stage 1 V2 — Vehicle Base Price Model

## Purpose

Stage 1 V2 is the production vehicle-value model that runs before Stage 2 and Stage 3. The existing V1 model remains unchanged as an automatic fallback.

Unlike the existing model, V2 deliberately excludes sale month and `year_month`. Stage 2 remains responsible for market movement and Stage 3 for seasonality.

## Architecture

- 50/50 ensemble of two XGBoost gradient-boosted tree models
- one component predicts the raw dollar price; one predicts the log-transformed price
- one-hot encoding for categorical values
- additional vehicle information: trim, transmission, state, exterior and interior color
- explicit make-model interaction selected on a separate validation split
- no VIN, seller or MMR feature; MMR was excluded to avoid target-like leakage
- architecture and ensemble weight selected without using the final test split

## Evaluation

> Preferred scientific comparison: `model_results_stage1_v1_v2_shared_split.md`.
> In that benchmark both models are retrained from scratch on the same split;
> V2 reduces MAE from $1,830.95 to $1,370.15 (25.17%).

- Rows: 529,169
- Train/test split: 423,335 / 105,834
- V2 MAE: **$1,370.15**
- V2 RMSE: **$2,400.34**
- V2 R²: **0.9366**
- V2 MAPE: **15.13%**
- Existing model MAE on exactly the same V2 test rows: **$1,831.39**
- MAE improvement: **$461.24 (25.19%)**

The ensemble is optimized for MAE in dollars. Compared with the single log model, it lowers MAE but can trade off some RMSE or percentage-error performance.

## Error by price segment

| Segment | Test rows | MAE | MAPE |
|---|---:|---:|---:|
| Budget | 17,773 | $868.55 | 42.20% |
| Economy | 21,689 | $1,076.31 | 14.62% |
| Mid-Range | 46,049 | $1,148.19 | 8.13% |
| Premium | 20,475 | $2,069.77 | 7.85% |
| Luxury | 1,780 | $7,581.99 | 13.35% |

## Reproduce

```bash
uv run python scripts/train_stage1_v2.py --max-rows 0
```

Omit `--max-rows 0` for a faster 200,000-row development run.
