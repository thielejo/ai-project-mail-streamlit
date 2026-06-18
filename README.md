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
app/                        Streamlit demo app
docs/                       Project proposal, session notes, data documentation
model_comparison/           Model benchmarking results (6 models compared)
models/                     Trained model files
notebooks/                  Exploratory notebooks
scripts/
  build_features.py         Feature engineering from cleaned CSV
  train_price_model.py      Train Stage 1 model (XGBoost with HistGB fallback)
  evaluate_segments.py      Segment error analysis on existing model
  compare_models.py         Full model comparison benchmark
  enrich_macro.py           Download FRED macro indicators → macro_index.csv
  train_stage1.py           Alternative XGBoost pipeline (Moritz branch)
car_prices_clean.csv        Cleaned Manheim auction data (558,743 rows, 2014–2015)
car_prices_features.csv     Engineered features ready for model training (534,318 rows)
macro_index.csv             FRED macro indicators 1996–2026-06 (CPI, rates, sentiment)
model_results.md            Latest Stage 1 evaluation results incl. segment breakdown
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

# 3. Run Streamlit demo
uv run streamlit run app/streamlit_app.py
```

---

## Model Performance (Stage 1)

Current best model: **HistGradientBoostingRegressor** (200,000 training rows)

| Metric | Model | Median baseline |
|---|---:|---:|
| MAE | $1,766 | $6,932 |
| RMSE | $3,259 | $9,705 |
| R² | 0.898 | −0.025 |
| MAPE | 12.6% | 49.5% |

Error by price segment: see [`model_results.md`](model_results.md)
Full model comparison (6 models): see [`model_comparison/model_comparison.md`](model_comparison/model_comparison.md)

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
