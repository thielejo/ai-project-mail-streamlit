# Stage 1 Model Results

This document summarizes the first vehicle price prediction model.

## Goal

The Stage 1 model predicts the baseline selling price from vehicle-level features only.

## Input Features

- `vehicle_age`
- `sale_month`
- `odometer`
- `condition`
- `year_month`
- `make`
- `model`
- `body`

Target variable:

- `sellingprice`

## Training Setup

- Input file: `car_prices_features.csv`
- Rows used for training/evaluation: 200,000
- Test split: 20%
- Test rows: 40,000
- Model: HistGradientBoostingRegressor
- Text features: encoded for model training
- Target transformation: `log1p(sellingprice)`
- Saved model: `models/price_model.joblib`

## Environment Note

XGBoost could not be loaded because the macOS OpenMP runtime `libomp.dylib` is missing. Install it with `brew install libomp` to enable XGBoost locally.

The script automatically used a sklearn gradient boosting model so the baseline can still be trained locally. Once the missing XGBoost system dependency is installed, the same script will use XGBoost.


## Results

| Metric | HistGradientBoostingRegressor | Median-price baseline |
|---|---:|---:|
| MAE | $1,849.96 | $6,932.41 |
| RMSE | $3,298.97 | $9,704.69 |
| R2 | 0.8816 | -0.0248 |
| MAPE | 16.42% | 121.05% |

## Most Important Features

| Feature | Importance |
|---|---:|
| make | 2470.4691 |
| vehicle_age | 2415.7414 |
| body | 2042.4342 |
| odometer | 1913.0556 |
| model | 1517.4431 |
| condition | 610.6118 |
| year_month | 32.523 |
| sale_month | 15.7077 |

## How to Reproduce

Run this command from the project root:

```bash
uv run python scripts/train_price_model.py
```

For full-dataset training, run:

```bash
uv run python scripts/train_price_model.py --max-rows 0
```
