# Project Theme: Hybrid AI Agent for Dynamic Used Car Pricing

# Project Proposal: Hybrid AI Agent for Dynamic Used Car Pricing

## Team Members of Team MAIL

| Name | GitHub Username | Email |
| :--- | :--- | :--- |
| Johanna Thiele | [@Thielejo](https://github.com/Thielejo) | thielejo@hs-pforzheim.de |
| Moritz Binder | [@Moritzb1](https://github.com/Moritzb1) | bindermo@hs-pforzheim.de |
| Pasca Muller | [@Paelus](https://github.com/Paelus) | muelle6p@hs-pforzheim.de |
| Tara Golle | [@trsphgll](https://github.com/trsphgll) | golletar@hs-pforzheim.de |

## Selected Track
**Hybrid Agent** - Building an AI agent that encapsulates a classic ML project as a functional tool.

---

## 🚀 Project Vision: "Universal Pricing & Strategy Agent"

### Core Objective
The goal of this project is to develop a B2B/B2C hybrid AI assistant that shifts used car valuation from a static snapshot into a dynamic, macro-aware profit optimization system. The agent determines a vehicle's tagesaktueller fair value and evaluates price trends over a 12-month horizon to deliver data-driven strategic timing recommendations (**"Best-Time-To-Sell"**).

### Data Strategy & Engineering
To avoid **Omitted Variable Bias**, retail dealer margin inflation, and cross-currency mismatches (e.g., mixing EUR/GBP with USD), the scope is strictly restricted to the US market. The architecture fuses two high-fidelity data layers:
* **Micro-Level (Vehicle Attributes):** Real transactional wholesale data from the **Manheim B2B Auction Dataset (550,000+ vehicles)**, providing unbiased transaction prices rather than retail "asking" prices.
* **Macro-Level (Economic Context):** Time-series data sourced from the US Federal Reserve Economic Data (FRED) and the **Manheim Used Vehicle Value Index (MUVVI)** to map market conditions, purchasing power elasticity, and historical inflation benchmarks.

---

## 🏛️ System Architecture: The 3-Stage Ensemble

Because individual vehicles lack multi-year longitudinal tracking within the dataset, the system bypasses this limitation via an innovative **Three-Stage Ensemble Architecture** that decouples physical wear-and-tear from macroeconomic volatility and temporal market psychology.

### 🧠 Stage 1: The Micro-Model (Vehicle Physics)
* **Technology:** `XGBoost Regressor` (Machine Learning).
* **Task:** This model handles high-dimensional cross-sectional features (Brand, Model, Body Type, Odometer Mileage, and Manheim Condition Score). Through feature engineering, it computes the exact chronological age at the time of sale (`vehicle_age`). 
* **Output:** The **Baseline Price** (The intrinsic physical value of the vehicle scaled to the neutral, unshocked economic baseline of 2015).

### 📈 Stage 2: The Macro-Model (The World Economy & MUVVI)
* **Technology:** Deterministic Financial Heuristic / Expert System.
* **Task:** Traditional ML regressions fail here due to massive **Structural Breaks** caused by "Black Swan" events (e.g., the 2020–2022 pandemic chip shortages where high interest rates anomaly-correlated with soaring car prices). To combat this, an expert system applies purchasing power elasticity and historical trends using the **Manheim Used Vehicle Value Index (MUVVI)** and FRED Consumer Price Index (`CUSR0000SETA01`). It evaluates base inflation against the chilling effects of the US Federal Funds Rate (`FEDFUNDS`).
* **Output:** The **Market Multiplier** (A dynamic scalar, e.g., `1.20x` or `0.95x`, adjusting the Stage 1 baseline price to current real-world economic conditions).

### 📅 Stage 3: The Strategy-Model (Micro-Seasonality)
* **Technology:** Rule-Based Domain Knowledge System.
* **Task:** This model captures localized, demand-side temporal psychology based on the vehicle's `body` type. Rather than applying a flat monthly average, it calculates specialized sub-trends:
    * *Convertibles & Sports Cars:* Subject to weather-driven emotional peaks in mid-summer (June/July) and heavy drops in winter.
    * *SUVs & 4x4s:* Driven by safety-conscious winter preparation, peaking late autumn (October/November).
    * *Budget Sedans & Compacts:* Heavily accelerated by the US **"Spring Bounce"** (February–April) when lower-income buyers receive liquid capital via annual tax refunds.
* **Output:** The **Seasonal Adjustment Factor** and an automated 12-month trajectory map.

---

## 💻 The End Product: Streamlit Hybrid UI & LLM Orchestration

The entire engine is compiled into an interactive **Streamlit Web Application** designed for rapid prototyping and enterprise assessment. 

1. **User Input:** The user inputs standard vehicle criteria and selects the current US economic environment (interest rate regime).
2. **Algorithmic Fusion:** The backend computes the ultimate price utilizing the core synthesis equation:
   $$	ext{Live Price} = 	ext{Baseline Price (Stage 1)} 	imes 	ext{Market Multiplier (Stage 2)} 	imes 	ext{Seasonal Factor (Stage 3)}$$
3. **LLM Orchestrator:** An integrated Large Language Model (via Ollama/Local API) processes these raw mathematical metrics into intuitive, natural language tactical advice (e.g., *"Based on the high Federal Funds Rate and the fact that your asset is a convertible, do not sell in November. Hold the vehicle for 5 months to leverage the Spring Bounce, maximizing your net return by an estimated \$1,800 despite the minor mileage depreciation"*).

---

## 📋 Expected Deliverables (by 31.07.2026)
- [ ] **Reproducible Pipeline:** Clean, fully modularized Python pipeline using `uv` for lightning-fast dependency management (`pyproject.toml` included) and Git version control.
- [ ] **Scientific Paper:** A comprehensive 12-page paper following the Springer **LNCS format**, documenting the architectural methodologies, feature engineering, baseline evaluation (MAE/RMSE), and economic profit simulations.
- [ ] **Presentation Deck:** A professional PDF presentation tracking the project lifecycle and architectural pivots.

---

## 📊 Data Sources

### Micro (Vehicle Properties)
* **Primary Source:** Manheim Used Car Auction Data (US B2B Transactions)
* [Kaggle: Used Car Auction Prices](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices)

### Macro (Economic Indicators)
* **Primary Source:** Federal Reserve Economic Data (FRED) & Cox Automotive
* [US Interest Rates (FEDFUNDS)](https://fred.stlouisfed.org/series/FEDFUNDS)
* [Used Car Price Index / MUVVI Proxy (CUSR0000SETA01)](https://fred.stlouisfed.org/series/CUSR0000SETA01)

---
*Proposal submitted for review. We look forward to feedback via GitHub issues.*
