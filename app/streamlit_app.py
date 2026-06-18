from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "car_prices_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "price_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "price_model_metrics.json"

FEATURE_COLUMNS = [
    "vehicle_age",
    "sale_month",
    "odometer",
    "condition",
    "year_month",
    "make",
    "model",
    "body",
]


st.set_page_config(
    page_title="Universal Pricing Agent",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_data() -> pd.DataFrame:
    columns = ["make", "model", "body", "sellingprice", "odometer", "condition"]
    return pd.read_csv(FEATURES_PATH, usecols=columns)


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}

    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def get_default_index(options: list[str], preferred_value: str) -> int:
    if preferred_value in options:
        return options.index(preferred_value)
    return 0


def build_prediction_input(
    vehicle_age: int,
    sale_month: int,
    odometer: int,
    condition: float,
    sale_year: int,
    make: str,
    model: str,
    body: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vehicle_age": vehicle_age,
                "sale_month": sale_month,
                "odometer": odometer,
                "condition": condition,
                "year_month": f"{sale_year}-{sale_month:02d}",
                "make": make,
                "model": model,
                "body": body,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


data = load_feature_data()
metrics = load_metrics()
model = load_model()

st.title("Universal Pricing Agent")
st.caption("Stage 1 Demo: Fahrzeugdaten eingeben und einen geschätzten Verkaufspreis berechnen.")

left_column, right_column = st.columns([0.95, 1.05], gap="large")

with left_column:
    st.subheader("Fahrzeugdaten")

    make_options = sorted(data["make"].dropna().unique())
    selected_make = st.selectbox(
        "Marke",
        make_options,
        index=get_default_index(make_options, "bmw"),
    )

    make_data = data[data["make"] == selected_make]
    model_options = sorted(make_data["model"].dropna().unique())
    if not model_options:
        model_options = sorted(data["model"].dropna().unique())

    selected_model = st.selectbox("Modell", model_options)

    body_options = sorted(data["body"].dropna().unique())
    selected_body = st.selectbox(
        "Karosserieform",
        body_options,
        index=get_default_index(body_options, "sedan"),
    )

    input_grid_left, input_grid_right = st.columns(2)

    with input_grid_left:
        model_year = st.number_input(
            "Baujahr",
            min_value=1990,
            max_value=2016,
            value=2014,
            step=1,
        )
        sale_year = st.number_input(
            "Verkaufsjahr",
            min_value=2014,
            max_value=2016,
            value=2015,
            step=1,
        )
        sale_month = st.select_slider(
            "Verkaufsmonat",
            options=list(range(1, 13)),
            value=1,
            format_func=lambda month: f"{month:02d}",
        )

    with input_grid_right:
        odometer = st.number_input(
            "Mileage / Odometer",
            min_value=1,
            max_value=500_000,
            value=50_000,
            step=1_000,
        )
        condition = st.slider(
            "Zustand",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
        )

    vehicle_age = max(int(sale_year) - int(model_year), 0)
    st.info(f"Berechnetes Fahrzeugalter: {vehicle_age} Jahre")

with right_column:
    st.subheader("Preisprognose")

    prediction_input = build_prediction_input(
        vehicle_age=vehicle_age,
        sale_month=int(sale_month),
        odometer=int(odometer),
        condition=float(condition),
        sale_year=int(sale_year),
        make=selected_make,
        model=selected_model,
        body=selected_body,
    )

    prediction = float(model.predict(prediction_input)[0])

    st.metric("Geschätzter Verkaufspreis", format_currency(prediction))

    mae = metrics.get("metrics", {}).get("mae")
    r2 = metrics.get("metrics", {}).get("r2")
    if mae is not None and r2 is not None:
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Durchschnittlicher Fehler", format_currency(float(mae)))
        metric_right.metric("R2 Score", f"{float(r2):.3f}")

    st.dataframe(
        prediction_input.rename(
            columns={
                "vehicle_age": "Alter",
                "sale_month": "Monat",
                "odometer": "Mileage",
                "condition": "Zustand",
                "year_month": "Jahr-Monat",
                "make": "Marke",
                "model": "Modell",
                "body": "Karosserie",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

summary_columns = st.columns(3)
summary_columns[0].metric("Trainingsdaten", f"{metrics.get('rows_used', 0):,}".replace(",", "."))
summary_columns[1].metric("Testdaten", f"{metrics.get('test_rows', 0):,}".replace(",", "."))
summary_columns[2].metric("Modell", metrics.get("model_name", "Preis-Modell"))

with st.expander("Was passiert hier Schritt fuer Schritt?"):
    st.markdown(
        """
1. Die App nimmt deine Fahrzeugdaten aus dem Formular.
2. Daraus wird dieselbe Tabellenstruktur gebaut, die auch beim Training verwendet wurde.
3. Das gespeicherte Modell aus `models/price_model.joblib` wird geladen.
4. Das Modell berechnet daraus einen Basispreis.
5. Dieser Basispreis ist Stage 1 eures Projekts. Markt- und Saisonfaktoren kommen spaeter dazu.
"""
    )

with st.expander("Wichtigste Einflussfaktoren aus dem Modelltest"):
    top_features = metrics.get("top_features", [])
    if top_features:
        st.dataframe(pd.DataFrame(top_features), use_container_width=True, hide_index=True)
    else:
        st.write("Noch keine Feature-Importance gespeichert.")

with st.expander("Modellgenauigkeit nach Preissegment"):
    segment_metrics = metrics.get("segment_metrics", [])
    if segment_metrics:
        df_seg = pd.DataFrame(segment_metrics).rename(
            columns={
                "segment": "Segment",
                "price_range": "Preisbereich",
                "n": "Testdaten",
                "mae": "MAE ($)",
                "rmse": "RMSE ($)",
                "mape_percent": "MAPE (%)",
            }
        )
        st.dataframe(df_seg, use_container_width=True, hide_index=True)
        st.caption(
            "Das Modell ist am genauesten im Mittelklasse-Segment ($10k–$20k). "
            "Budget-Fahrzeuge (MAPE ~35%) und Luxusfahrzeuge (MAPE ~21%) sind schwerer vorherzusagen."
        )
    else:
        st.write("Segmentauswertung noch nicht verfügbar.")
