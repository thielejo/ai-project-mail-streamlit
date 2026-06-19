# Feature Engineering: Car Price Model Dataset

This document summarizes the feature engineering step after cleaning the Manheim car prices dataset.

## Input

- Source file: `car_prices_clean.csv`
- Input rows: 558,743
- Script: `scripts/build_features.py`

## Output

- Feature file: `car_prices_features.csv`
- Output rows: 534,318
- Removed rows: 24,425

Rows were removed if one of the required modeling fields was missing.

## Features Created

| Feature | Meaning |
|---|---|
| `vehicle_age` | Age of the vehicle at sale time. Negative values are clipped to `0`, because next-year model cars can be sold in the previous calendar year. |
| `year_month` | Sale month in `YYYY-MM` format. |
| `sale_month` | Sale month as number from `1` to `12`. |
| `make` | Vehicle make, for example `kia`, `bmw`, `volvo`. |
| `model` | Vehicle model, for example `sorento`, `3 series`, `s60`. |
| `odometer` | Mileage at sale time. |
| `condition` | Manheim vehicle condition score. |
| `body` | Body type, for example `suv` or `sedan`. |
| `sellingprice` | Target variable for model training. |

## How to Reproduce

Run this command from the project root:

```bash
uv run python scripts/build_features.py
```

This regenerates `car_prices_features.csv`.
