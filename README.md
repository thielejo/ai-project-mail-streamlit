[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NELeUmAZ)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=23211453)

# Universal Pricing Agent - BIS5522 AI & Machine Learning

**Team MAIL:** Johanna Thiele · Moritz Binder · Pascal Müller · Tara Golle  
**Deadline:** 31.07.2026  
**Live Demo:** [https://ai-project-mail.streamlit.app](https://ai-project-mail.streamlit.app)

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
The current app uses the Stage-1 V2 model with CPI and seasonal adjustments.
Separately, the strongest documented Stage-1 candidate is a tuned CatBoost model
with displacement data; it is documented but not yet integrated into the app.

## Current Status & Results

### Stage 1 - Vehicle-Value Model

The app currently uses the Stage-1 V2 XGBoost ensemble as its production model.
On a strict shared train/test split, V2 improved MAE from **$1,830.95** to
**$1,370.15** compared with the previous V1 model.

| Metric | V2 XGBoost Ensemble | Previous V1 |
|---|---:|---:|
| MAE | $1,370.15 | $1,830.95 |
| RMSE | $2,400.34 | $3,276.81 |
| R² | 0.9366 | 0.8818 |
| MAPE | 15.13% | 16.45% |

The strongest Stage-1 candidate so far is a tuned **CatBoost + displacement**
model with **MAE $1,042**, **RMSE $1,875**, **R² 0.961**, and **MAPE 11.8%**.
This model is documented as the target candidate, but still needs integration
with the app and re-evaluation for Stage 2 and Stage 3.

Details:

- Current Stage-1 model note: [`docs/stage1/stage1_current_model.md`](docs/stage1/stage1_current_model.md)
- V1/V2 comparison: [`docs/stage1/stage1_model_comparison.md`](docs/stage1/stage1_model_comparison.md)
- CatBoost tuning: [`docs/stage1/stage1_tuning_results.md`](docs/stage1/stage1_tuning_results.md)
- Model benchmark: [`docs/stage1/stage1_benchmark_models.md`](docs/stage1/stage1_benchmark_models.md)

### Stage 2 - CPI Macro Adjustment

Stage 2 adjusts the Stage-1 baseline price with a used-car CPI multiplier from
FRED. The historical 2014-2015 test period is close to the CPI baseline, so the
backtest changes accuracy only slightly. Its main purpose is forward projection
to later market price levels.

| Date | CPI Multiplier | Effect |
|---|---:|---|
| 2015-01 | 1.0000 | baseline |
| 2021-12 | 1.1396 | +14.0% vs. baseline |
| 2023-09 | 1.2200 | +22.0% vs. baseline |
| 2026-06 | 1.2177 | +21.8% vs. baseline |

Backtest on the V2 setup: MAE **$1,370.16 -> $1,376.22** after CPI adjustment
(+0.44%). See [`docs/stage2/model_results_stage2.md`](docs/stage2/model_results_stage2.md).

### Stage 3 - Seasonal Adjustment

Stage 3 adds a rule-based seasonal factor by body type and sale month. The
factors are generated from CPI-normalized Stage-1 residuals and smoothed toward
neutral when data is sparse.

On the separated rule holdout, Stage 3 improves MAE from **$1,353.15** to
**$1,339.84** (-0.98%). Months without historical observations remain neutral
at 1.0. See [`docs/stage3/model_results_stage3.md`](docs/stage3/model_results_stage3.md).

## Repository Structure

```
app/                  Streamlit demo app
archive/              Historical handoffs, deprecated notes, archived notebooks
docs/                 Project documentation and model reports
exploration/          Exploratory analysis notebooks
model_comparison/     Machine-readable model benchmark results
models/               Trained model files and evaluation outputs
scripts/              Data preparation, training, evaluation, Stage 2/3 modules
tuning_experiment/    CatBoost tuning experiments
vin_fin_enrichment/   VIN/FIN displacement enrichment experiments

Aktueller_Stand.md    Internal project-status context
car_prices_clean.csv  Cleaned Manheim auction data
car_prices_features.csv
                      Feature-engineered training data
macro_index.csv       FRED macro indicators
```

`car_prices_macro.csv` is intentionally gitignored because of its file size. It
can be regenerated with:

```bash
uv run python scripts/enrich_macro.py
```

## Quickstart

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Git.

```bash
git clone https://github.com/digital-business-lectures/ai-project-mail.git
cd ai-project-mail
uv sync

# Build feature dataset
uv run python scripts/build_features.py

# Train the Stage-1 production model used by the app
uv run python scripts/train_stage1_production.py --max-rows 0

# Evaluate Stage 2 and Stage 3
uv run python scripts/evaluate_stage2.py
uv run python scripts/evaluate_stage3.py

# Run the demo app
uv run streamlit run app/streamlit_app.py
```

## Data Sources

- **Micro data:** Manheim Used Car Auction Data via [Kaggle](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices), 558,743 US wholesale transactions from 2014-2015.
- **Macro data:** Federal Reserve Economic Data (FRED), especially the used-car CPI series used for Stage 2.
- **VIN/FIN enrichment:** NHTSA vPIC API for displacement information in the Stage-1 enrichment experiments.

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

- [`Aktueller_Stand.md`](Aktueller_Stand.md) - detailed current project context
- [`HANDOFF.md`](HANDOFF.md) - handoff notes and remaining work
- [`docs/data_cleaning.md`](docs/data_cleaning.md) - data-cleaning decisions
- [`docs/feature_engineering.md`](docs/feature_engineering.md) - feature-engineering decisions
- [`docs/project_proposal.md`](docs/project_proposal.md) - original historical proposal
