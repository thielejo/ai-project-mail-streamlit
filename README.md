# Universal Pricing Agent

Streamlit demo for estimating used-car sale prices with a trained machine learning model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Streamlit Cloud

- Repository: `thielejo/ai-project-mail-streamlit`
- Branch: `main`
- Main file path: `app/streamlit_app.py`
- Secrets: none required

## Included artifacts

- `app/streamlit_app.py`: Streamlit user interface
- `models/price_model.joblib`: trained price model
- `models/price_model_metrics.json`: model metrics and feature importance
- `car_prices_features.csv`: feature data used by the app
