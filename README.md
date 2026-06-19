[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NELeUmAZ)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=23211453)

# Universal Pricing Agent — BIS5522 AI & Machine Learning

**Team MAIL:** Johanna Thiele · Moritz Binder · Pascal Müller · Tara Golle
**Deadline:** 31.07.2026

A hybrid AI agent for dynamic used car pricing. The system combines a machine learning baseline model with macroeconomic adjustment and seasonal rules to produce a current, market-aware price estimate for any vehicle.

---

## Architecture

```
Stage 1 (Micro)     XGBoost/HistGB on vehicle attributes → Baseline price (2015 USD)
Stage 2 (Macro)     CPI multiplier from FRED data       → Inflation-adjusted live price
Stage 3 (Seasonal)  Rule-based by body type & month     → Best-time-to-sell advice
```

**End product:** Streamlit app + LLM orchestration layer

---

## Repository Structure

```
app/                        Streamlit demo app (Stage 1 + Stage 2)
docs/                       Project proposal, session notes, data documentation
model_comparison/           Model benchmarking results (6 models compared)
models/                     Trained model files and evaluation results
notebooks/                  Exploratory notebooks
scripts/
  build_features.py         Feature engineering from cleaned CSV
  train_price_model.py      Train Stage 1 model (XGBoost with HistGB fallback)
  stage2_macro.py           Stage 2 module: CPI lookup and price adjustment
  evaluate_stage2.py        Stage 2 backtest and forward projection
  evaluate_segments.py      Segment error analysis on existing model
  compare_models.py         Full model comparison benchmark
  enrich_macro.py           Download FRED macro indicators → macro_index.csv
  train_stage1.py           Alternative XGBoost pipeline (older)
Aktueller_Stand.md          Full project context for AI assistants (start here)
car_prices_clean.csv        Cleaned Manheim auction data (558,743 rows, 2014–2015)
car_prices_features.csv     Engineered features ready for model training (534,318 rows)
macro_index.csv             FRED macro indicators 1996–2026-06 (CPI, rates, sentiment)
model_results.md            Stage 1 evaluation results incl. segment breakdown
model_results_stage2.md     Stage 2 evaluation: backtest and forward projections
```

> `car_prices_macro.csv` (98 MB) is gitignored. Regenerate with:
> `uv run python scripts/enrich_macro.py`

---

## Quickstart

```bash
# Install dependencies
uv sync

# 1. Build feature dataset (from car_prices_clean.csv)
uv run python scripts/build_features.py

# 2. Train Stage 1 model
uv run python scripts/train_price_model.py

# 3. Evaluate Stage 2
uv run python scripts/evaluate_stage2.py

# 4. Run Streamlit demo
uv run streamlit run app/streamlit_app.py
```

---

## Model Performance

### Stage 1 — HistGradientBoostingRegressor (200,000 training rows)

| Metric | Model | Median baseline |
|---|---:|---:|
| MAE | $1,850 | $6,932 |
| RMSE | $3,299 | $9,705 |
| R² | 0.882 | −0.025 |
| MAPE | 16.4% | 121.1% |

Error by price segment: see [`model_results.md`](model_results.md)
Full model comparison (6 models): see [`model_comparison/model_comparison.md`](model_comparison/model_comparison.md)

### Stage 2 — CPI Macro Adjustment

| Date | CPI Multiplier | Effect |
|---|---:|---|
| 2015-01 (baseline) | 1.0000 | Reference |
| 2021-12 (COVID surge) | 1.1396 | +14% vs. baseline |
| 2023-09 (all-time high) | 1.2200 | +22% vs. baseline |
| 2026-06 (current) | 1.2177 | +21.8% vs. baseline |

Backtest on historical test set: Δ MAE = −$0.15 (0.01%) — does not hurt accuracy on 2014–2015 data.
See [`model_results_stage2.md`](model_results_stage2.md)

---

## Data Sources

- **Micro:** Manheim Used Car Auction Data via [Kaggle](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices) — 558,743 US wholesale transactions, 2014–2015
- **Macro:** Federal Reserve Economic Data (FRED) — CPI, Federal Funds Rate, Unemployment, Consumer Sentiment, Recession indicator

---

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), Git.

```bash
git clone https://github.com/digital-business-lectures/ai-project-mail.git
cd ai-project-mail
uv sync
```
