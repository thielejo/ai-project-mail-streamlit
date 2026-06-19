# Model Comparison: Stage 1 Price Prediction

## Methodology

All models are evaluated under identical conditions:

- **Dataset:** `car_prices_features.csv` — 200,000 rows sampled (random_state=42)
- **Split:** 80/20 train/test, random_state=42
- **Features:** `vehicle_age`, `sale_month`, `odometer`, `condition`, `year_month`, `make`, `model`, `body`
- **Target:** `sellingprice` — log1p-transformed before training, expm1 after prediction
- **Preprocessing:** StandardScaler on numerics, OneHotEncoder on categoricals (same pipeline for all models)
- **Baseline:** Median selling price of the training set — predicts the same value for every car

Segment boundaries: Budget (<$5k), Economy ($5k–$10k), Mid-Range ($10k–$20k), Premium ($20k–$40k), Luxury (>$40k)

---

## Overall Results

| Model | MAE | MAE % | RMSE | R² | MAPE | Train time |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | $1,766 | 12.6% | $3,057 | 0.8983 | 16.0% | 33.2s |
| XGBoost | $2,055 | 14.7% | $3,521 | 0.8651 | 18.2% | 1.4s |
| Linear Regression | $2,193 | 15.7% | $3,691 | 0.8518 | 20.1% | 0.8s |
| Ridge Regression | $2,202 | 15.7% | $3,711 | 0.8502 | 20.2% | 0.5s |
| Random Forest | $2,333 | 16.6% | $3,839 | 0.8397 | 21.2% | 80.5s |
| Median Baseline | $6,932 | 49.5% | $9,705 | -0.0248 | 121.0% | 0.0s |
| Lasso Regression | $7,132 | 50.9% | $10,168 | -0.1249 | 104.5% | 0.4s |

---

## Error by Price Segment

### HistGradientBoosting

| Segment | Price Range | N | MAE | MAPE |
|---|---|---:|---:|---:|
| Budget | $500–$5,000 | 6,645 | $774 | 35.5% |
| Economy | $5,000–$10,000 | 8,064 | $1,166 | 15.7% |
| Mid-Range | $10,000–$20,000 | 17,410 | $1,536 | 10.7% |
| Premium | $20,000–$40,000 | 7,862 | $2,949 | 11.3% |
| Luxury | $40,000–$150,000 | 699 | $10,323 | 18.9% |

### XGBoost

| Segment | Price Range | N | MAE | MAPE |
|---|---|---:|---:|---:|
| Budget | $500–$5,000 | 6,645 | $844 | 39.7% |
| Economy | $5,000–$10,000 | 8,064 | $1,324 | 17.6% |
| Mid-Range | $10,000–$20,000 | 17,410 | $1,733 | 12.2% |
| Premium | $20,000–$40,000 | 7,862 | $3,493 | 13.2% |
| Luxury | $40,000–$150,000 | 699 | $13,499 | 24.9% |

### Linear Regression

| Segment | Price Range | N | MAE | MAPE |
|---|---|---:|---:|---:|
| Budget | $500–$5,000 | 6,645 | $886 | 46.6% |
| Economy | $5,000–$10,000 | 8,064 | $1,249 | 16.6% |
| Mid-Range | $10,000–$20,000 | 17,410 | $1,969 | 13.7% |
| Premium | $20,000–$40,000 | 7,862 | $4,043 | 15.4% |
| Luxury | $40,000–$150,000 | 699 | $9,758 | 17.9% |

### Ridge Regression

| Segment | Price Range | N | MAE | MAPE |
|---|---|---:|---:|---:|
| Budget | $500–$5,000 | 6,645 | $881 | 46.5% |
| Economy | $5,000–$10,000 | 8,064 | $1,251 | 16.6% |
| Mid-Range | $10,000–$20,000 | 17,410 | $1,981 | 13.8% |
| Premium | $20,000–$40,000 | 7,862 | $4,050 | 15.5% |
| Luxury | $40,000–$150,000 | 699 | $9,972 | 18.1% |

### Random Forest

| Segment | Price Range | N | MAE | MAPE |
|---|---|---:|---:|---:|
| Budget | $500–$5,000 | 6,645 | $999 | 45.9% |
| Economy | $5,000–$10,000 | 8,064 | $1,622 | 21.5% |
| Mid-Range | $10,000–$20,000 | 17,410 | $2,039 | 14.4% |
| Premium | $20,000–$40,000 | 7,862 | $3,976 | 14.9% |
| Luxury | $40,000–$150,000 | 699 | $11,622 | 22.0% |

### Lasso Regression

| Segment | Price Range | N | MAE | MAPE |
|---|---|---:|---:|---:|
| Budget | $500–$5,000 | 6,645 | $7,854 | 444.1% |
| Economy | $5,000–$10,000 | 8,064 | $2,899 | 43.4% |
| Mid-Range | $10,000–$20,000 | 17,410 | $3,704 | 23.2% |
| Premium | $20,000–$40,000 | 7,862 | $15,286 | 57.7% |
| Luxury | $40,000–$150,000 | 699 | $40,673 | 78.3% |

---

## Model Descriptions & Hyperparameters

**HistGradientBoosting:** Sequentielles Boosting (sklearn). Schnell, keine XGBoost-Abhängigkeit. Aktuell in der Streamlit-App.

**XGBoost:** Sequentielles Boosting mit nativer XGBoost-Bibliothek (300 Bäume, lr=0.06, depth=6). Empfohlenes Modell.

**Linear Regression:** Einfachstes lineares Modell ohne Regularisierung. Dient als untere Vergleichsschwelle.

**Ridge Regression:** Linear mit L2-Regularisierung (alpha=10). Robuster bei korrelierten Features.

**Random Forest:** Ensemble aus 200 unabhängigen Entscheidungsbäumen (max_depth=20). Parallelisiert, robust gegen Overfitting.

**Lasso Regression:** Linear mit L1-Regularisierung (alpha=1). Setzt irrelevante Features auf 0.

---

## Conclusion

**HistGradientBoosting achieves the lowest MAE ($1,766) in this comparison** and is the recommended model for Stage 1.

**XGBoost** is a close second (MAE $2,055) and trains significantly faster (1.4s vs 33.2s), which may matter for CI/CD pipelines.

**Key findings:**

1. **Gradient Boosting variants dominate** — both HistGB and XGBoost outperform Random Forest and linear models.
2. **Linear Regression is surprisingly competitive** — its MAE is only ~25% worse than the winner, confirming that the feature engineering (vehicle_age, OHE for make/model) carries substantial signal.
3. **Random Forest underperforms despite long training time** (80s) — not recommended for this use case.
4. **Lasso Regression collapses** — alpha=1.0 is far too aggressive for the log-transformed target with high-cardinality OHE features; it over-regularizes and produces near-median predictions.

**Limitations of this comparison:**

- Data covers only 2014–2015 US wholesale auction prices (Manheim). Retail or European markets may differ.
- 200,000 rows were sampled for speed; full-dataset training (534k rows) may further improve all tree-based models.
- Hyperparameters were not tuned via cross-validation — a grid search over XGBoost `max_depth` and `n_estimators` could close the gap to HistGB.
- `sale_month` and `year_month` have low feature importance — seasonal effects are weak in this 2014–2015 dataset alone. Stage 2 (CPI multiplier) and Stage 3 (seasonal rules) address this by adding external economic signal.

## How to Reproduce

```bash
uv run python scripts/compare_models.py
```