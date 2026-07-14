[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NELeUmAZ)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=23211453)

# PricePilot - Universal Used-Car Pricing Agent

**Team MAIL:** Johanna Thiele · Moritz Binder · Pascal Müller · Tara Golle  
**Deadline:** 31.07.2026  
**Live Demo:** [Open PricePilot in Streamlit](https://ai-project-mail.streamlit.app)

## Project Overview

This project develops a hybrid AI agent for dynamic used-car pricing. Instead of
predicting a static vehicle value only from car attributes, the system combines
three perspectives: the vehicle itself, the broader used-car market level, and
seasonal demand effects.

The core idea is to estimate a time-neutral baseline price for a vehicle and
then adjust it to a target market context. This allows the app to answer a more
realistic pricing question: What is this car worth under current market
conditions, and how might timing affect the final price?

The project uses US wholesale auction transactions from the Manheim used-car
dataset as the micro-level data foundation and FRED macroeconomic indicators for
market-level price adjustment. The original project proposal is kept as a
historical submission in [`docs/project_proposal.md`](docs/project_proposal.md).

## How It Works

```
Stage 1 (Micro)     Vehicle-value ML model          -> time-neutral baseline price
Stage 2 (Macro)     CPI multiplier from FRED data   -> market-level price adjustment
Stage 3 (Seasonal)  Body type and month rules       -> seasonal fine-tuning
```

The Streamlit demo combines these stages into an interactive pricing interface.
The current app uses the tuned CatBoost production model with displacement,
followed by CPI and seasonal adjustments.

## Current Status & Results

### Stage 1 - Vehicle-Value Model

The production model is a tuned **CatBoost model with displacement** derived
from VIN data. It uses a log-transformed target and 14 vehicle features while
excluding identifiers and `MMR` to avoid leakage. The committed 2,000-tree
artifact stays below GitHub's 100 MB file limit and is loaded directly by the
app.

| Metric | CatBoost Production | Previous V2 XGBoost |
|---|---:|---:|
| MAE | **$1,056.54** | $1,370.15 |
| RMSE | **$1,892.82** | $2,400.34 |
| R² | **0.9606** | 0.9366 |
| MAPE | **11.91%** | 15.13% |

The tuning experiments reached MAE **$1,042** with a larger 3,000-tree model.
The deployed model trades only about $14 MAE for a substantially smaller,
committable artifact. The app automatically suggests displacement from a
make/model lookup and allows the value to be adjusted.

Details:

- Model results overview: [`docs/model_results.md`](docs/model_results.md)
- Current Stage-1 model note: [`docs/stage1/stage1_current_model.md`](docs/stage1/stage1_current_model.md)
- V1/V2 comparison: [`docs/stage1/stage1_model_comparison.md`](docs/stage1/stage1_model_comparison.md)
- CatBoost tuning: [`docs/stage1/stage1_tuning_results.md`](docs/stage1/stage1_tuning_results.md)
- Model benchmark: [`docs/stage1/stage1_benchmark_models.md`](docs/stage1/stage1_benchmark_models.md)

### Stage 2 - CPI Macro Adjustment

Stage 2 adjusts the Stage-1 baseline price with the FRED Used Cars and Trucks
CPI series (`CUSR0000SETA02`). The multiplier is normalized to the 2015 annual
average. The historical 2014-2015 test period is close to this baseline, so the
backtest changes accuracy only slightly. Its main purpose is forward projection
to later market price levels.

| Date | CPI Multiplier | Effect |
|---|---:|---|
| 2015 average | 1.0000 | normalization baseline |
| 2021-12 | 1.4378 | +43.8% vs. baseline |
| 2023-09 | 1.2743 | +27.4% vs. baseline |
| 2026-06 | 1.2224 | +22.2% vs. baseline |

The backtest runs against the **CatBoost production model** (the same model the
app serves): MAE **$1,056.54 -> $1,075.86** after CPI adjustment (+1.83%).
Because the 2014-2015 test period is itself close to the reference level, the
multipliers stay near 1.0 and this result mainly checks the adjustment
architecture rather than improving accuracy. Its value lies in forward
projection. See [`docs/stage2/model_results_stage2.md`](docs/stage2/model_results_stage2.md).

### Stage 3 - Seasonal Adjustment

Stage 3 adds a rule-based seasonal factor by body type and sale month. The
factors are generated from CPI-normalized Stage-1 residuals and smoothed toward
neutral when data is sparse.

The seasonal factors are generated against the current CatBoost production
baseline. On the separated rule holdout, Stage 3 improves MAE on CPI-normalized
prices from **$1,020.36** to **$991.24** (-2.85%). This tests the seasonal rule,
not a new independent Stage-1 model. Months without historical observations
remain neutral at 1.0. See
[`docs/stage3/model_results_stage3.md`](docs/stage3/model_results_stage3.md).

## Repository Structure

```
app/                  Streamlit demo app
archive/              Historical handoffs, deprecated notes, archived notebooks
data/                 Raw, cleaned, feature-engineered, and macro datasets
docs/                 Project documentation, model reports, and presentation slides
exploration/          Exploratory analysis notebooks
model_comparison/     Machine-readable model benchmark results
models/               Trained model files and evaluation outputs
scripts/              Data preparation, training, evaluation, Stage 2/3 modules
tuning/               CatBoost tuning and Stage-2/3 re-evaluation
vin_fin_enrichment/   VIN decoding pipeline and versioned displacement cache
```

Important data files:

| File | Purpose | Tracked |
|---|---|---|
| `data/car_prices.csv` | Original Manheim auction data | yes |
| `data/car_prices_clean.csv` | Cleaned source data | yes |
| `data/car_prices_features.csv` | Feature and app comparison data | yes |
| `data/macro_index.csv` | FRED indicators and CPI multiplier | yes |
| `data/car_prices_macro.csv` | Generated micro/macro merge | no |
| `vin_fin_enrichment/vin_decoded_cache_full.csv` | Versioned NHTSA VIN cache for reproducible displacement enrichment | yes |

The generated micro/macro merge is intentionally gitignored and can be rebuilt
from the tracked raw data and external FRED sources. The complete VIN cache is
versioned so that the production model can be reproduced without decoding every
VIN again.

```bash
uv run python scripts/enrich_macro.py
```

## Quickstart

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Git.

```bash
git clone https://github.com/digital-business-lectures/ai-project-mail.git
cd ai-project-mail
uv sync

# Clean the tracked raw dataset and build features
uv run python scripts/clean_car_prices.py
uv run python scripts/build_features.py

# Optional: regenerate macro data
uv run python scripts/enrich_macro.py

# Run the demo app with the committed production artifacts
uv run streamlit run app/streamlit_app.py
```

The versioned VIN decode cache is already included in a normal clone, so the
current CatBoost model can be rebuilt directly:

```bash
uv run python scripts/train_stage1_catboost.py --max-rows 0

# Recreate the current CatBoost-based Stage-2 and Stage-3 evaluations
uv run python scripts/evaluate_stage2.py
uv run python scripts/evaluate_stage3.py
```

To refresh or recreate the VIN cache from the free NHTSA API, run the following
resume-safe command. A full rebuild can take several hours:

```bash
uv run python vin_fin_enrichment/build_full_vin_cache.py
```

The current CatBoost-based Stage-2 and Stage-3 evaluation outputs are stored in
`models/stage2_evaluation.json`, `models/stage3_evaluation.json`, and
`models/stage3_seasonality_factors.csv`; the corresponding readable reports are
under `docs/stage2/` and `docs/stage3/`.

## Data Sources

- **Micro data:** Manheim Used Car Auction Data via [Kaggle](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices), 558,837 raw US wholesale auction rows from 2014-2015; 558,743 remain after removing rows without price or mileage.
- **Macro data:** Federal Reserve Economic Data (FRED), especially the used-car CPI series used for Stage 2.
- **VIN/FIN enrichment:** NHTSA vPIC API for the displacement information used by the Stage-1 production model; the decoded VIN cache is versioned for reproducibility.

## AI Usage

AI tools were used as supporting assistants during the project, especially for
Python implementation, debugging, repository refactoring, documentation drafts,
and technical review. The team made the project decisions, evaluated the
results, and decided which AI-assisted outputs were kept, changed, or rejected.

In practice, this means that ChatGPT/Codex and Claude/Claude Code accelerated
technical work and first drafts, but they did not replace the team's own
understanding and responsibility. Examples include the manual decision to avoid
data leakage through `MMR`, VIN, and seller identifiers, the correction of
initial FIN-effect interpretations after larger tests, and the final separation
of model improvements from macro and seasonal adjustments.

## Further Documentation

- [`docs/data_cleaning.md`](docs/data_cleaning.md) - data-cleaning decisions
- [`docs/feature_engineering.md`](docs/feature_engineering.md) - feature-engineering decisions
- [`docs/project_proposal.md`](docs/project_proposal.md) - original historical proposal
