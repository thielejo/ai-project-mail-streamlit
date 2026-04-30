# Project Proposal: Hybrid AI Agent for Dynamic Used Car Pricing

## Team Members of Team MAIL

| Name | GitHub Username | Email |
| --- | --- | --- |
| Johanna Thiele | @Thielejo | thielejo@hs-pforzheim.de |
| Moritz Binder | @Moritzb1 | bindermo@hs-pforzheim.de |
| Pasca Muller | @Paelus | muelle6p@hs-pforzheim.de |
| Tara Golle | @trsphgll | golletar@hs-pforzheim.de |

## Selected Track

Hybrid Agent - Building an AI agent that encapsulates a classic ML project as a functional tool.

## Project Vision

Development of a "Universal-Pricing-Agent" that transforms static used car valuation into a dynamic profit optimization problem. The agent forecasts prices over a 12-month horizon to identify the mathematically optimal selling time, maximizing net profit by balancing predicted price against holding costs.

## Data Strategy

We will fuse two distinct data dimensions to ensure real-world complexity:

- Micro-Level (Vehicle): High-dimensional features from real-world used car datasets (e.g., age, mileage, brand, fuel type, equipment, vehicle type).
- Macro-Level (Economic Context): Historical time-series data from external sources:
  - ECB interest rates and inflation data
  - Fuel price indices (ADAC/Statista)
  - Seasonal patterns (e.g., convertible demand in spring)

## Technical Methodology

1. ML Core:
   - Ensemble models (XGBoost/Random Forest) for non-linear interaction modeling.
   - Strict Time-Series Split for training/testing to prevent data leakage.
   - Evaluation metrics: RMSE, MAE, and economic profit simulation.

2. Hybrid Agent Architecture:
   - Backend: Python-based ML pipeline (scikit-learn, pandas, uv for dependency management).
   - Interface: Streamlit web app for intuitive user interaction.
   - LLM Orchestrator: Integration of an LLM to translate user input into structured queries and convert model outputs into strategic natural language recommendations (e.g., "Hold vehicle for 3 months").

3. Reproducibility:
   - Full documentation via Jupyter Notebooks / modular Python scripts.
   - Environment management via `uv` (`pyproject.toml` included).
   - All code, data preprocessing steps, and experiments version-controlled in this repository.

## Expected Deliverables (by 31.07.2026)

- [ ] Fully documented, reproducible code pipeline in the main branch.
- [ ] Scientific paper (ca. 12 pages, LNCS format) detailing research question, methodology, experiments, and results.
- [ ] Presentation deck (PDF) summarizing the project outcome.

## Risk Mitigation

- Data Availability: Backup datasets identified (e.g., Kaggle AutoML datasets enriched with public economic indicators).
- Scope Creep: Focus on MVP (single vehicle category) before scaling to multi-category support.
- LLM Costs: Use of local/open-source models (e.g., Llama 3 via Ollama) or capped API budgets.

## Data Sets

### Micro

- https://www.kaggle.com/code/georgeamadeus/car-sales-dataset
- https://www.kaggle.com/datasets/rebrowser/autotrader-dataset
- https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
- https://www.kaggle.com/datasets/tunguz/used-car-auction-prices

### Macro

- https://data.ecb.europa.eu/#dashboard-tab-3
- https://ec.europa.eu/eurostat/web/hicp/database
- https://fred.stlouisfed.org/series/ECBMRRFR

*Proposal submitted for review. We look forward to feedback via GitHub issues.*
