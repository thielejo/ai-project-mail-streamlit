from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from stage2_macro import (  # noqa: E402
    MACRO_SIGNAL_LABELS,
    apply_stage2,
    get_macro_context,
    load_macro_index,
)
from stage3_seasonality import (  # noqa: E402
    apply_stage3,
    load_seasonality_factors,
)
from stage1_runtime import (  # noqa: E402
    V1_METRICS_PATH,
    V2_METRICS_PATH,
    build_v1_input,
    build_v2_input,
    get_v2_category_options,
    load_production_model,
)

FEATURES_PATH = PROJECT_ROOT / "car_prices_features.csv"
STAGE2_EVAL_PATH = PROJECT_ROOT / "models" / "stage2_evaluation.json"
MACRO_PATH = PROJECT_ROOT / "macro_index.csv"
SEASONALITY_V1_PATH = PROJECT_ROOT / "models" / "seasonality_factors.csv"
SEASONALITY_V2_PATH = PROJECT_ROOT / "models" / "seasonality_factors_v2.csv"

MACRO_AVAILABLE_YEARS = list(range(1996, 2027))
MACRO_AVAILABLE_MONTHS = list(range(1, 13))
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez",
}


st.set_page_config(page_title="Universal Pricing Agent", layout="wide")


@st.cache_resource
def load_model():
    return load_production_model()


@st.cache_data
def load_feature_data() -> pd.DataFrame:
    return pd.read_csv(
        FEATURES_PATH, usecols=["make", "model", "body", "sellingprice", "odometer", "condition"]
    )


@st.cache_data
def load_metrics(model_version: str) -> dict:
    path = V2_METRICS_PATH if model_version == "v2" else V1_METRICS_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_stage2_eval() -> dict:
    if not STAGE2_EVAL_PATH.exists():
        return {}
    return json.loads(STAGE2_EVAL_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_macro() -> pd.DataFrame:
    return load_macro_index(MACRO_PATH)


@st.cache_data
def load_seasonality(model_version: str) -> pd.DataFrame:
    path = SEASONALITY_V2_PATH if model_version == "v2" else SEASONALITY_V1_PATH
    return load_seasonality_factors(path)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def get_default_index(options: list, preferred) -> int:
    if preferred in options:
        return options.index(preferred)
    return 0


data = load_feature_data()
model, model_version = load_model()
metrics = load_metrics(model_version)
stage2_eval = load_stage2_eval()
macro = load_macro()
seasonality = load_seasonality(model_version)
v2_options = get_v2_category_options(model) if model_version == "v2" else {}

st.title("Universal Pricing Agent")
st.caption(
    "Dreistufige Preisschätzung: Fahrzeugwert, aktuelles Marktpreisniveau und "
    "eine vorsichtige saisonale Anpassung."
)

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

    if model_version == "v2":
        st.caption("Stage 1 V2 nutzt zusätzliche Fahrzeugdetails für eine genauere Schätzung.")
        trim = st.selectbox(
            "Ausstattung (Trim)", v2_options["trim"],
            index=get_default_index(v2_options["trim"], "base"),
        )
        detail_left, detail_right = st.columns(2)
        with detail_left:
            transmission = st.selectbox(
                "Getriebe", v2_options["transmission"],
                index=get_default_index(v2_options["transmission"], "automatic"),
            )
            state = st.selectbox(
                "US-Bundesstaat", v2_options["state"],
                index=get_default_index(v2_options["state"], "ca"),
            )
        with detail_right:
            color = st.selectbox(
                "Außenfarbe", v2_options["color"],
                index=get_default_index(v2_options["color"], "black"),
            )
            interior = st.selectbox(
                "Innenfarbe", v2_options["interior"],
                index=get_default_index(v2_options["interior"], "black"),
            )
    else:
        trim = transmission = state = color = interior = ""

    input_left, input_right = st.columns(2)
    with input_left:
        model_year = st.number_input("Baujahr", min_value=1990, max_value=2022, value=2012, step=1)
        odometer = st.number_input(
            "Kilometerstand (Meilen)",
            min_value=1,
            max_value=500_000,
            value=50_000,
            step=1_000,
        )

    with input_right:
        condition = st.slider("Zustand", min_value=1.0, max_value=5.0, value=3.5, step=0.1)

    st.divider()
    st.subheader("Bewertungsdatum")
    st.caption("Für welchen Zeitpunkt soll der Marktpreis berechnet werden?")

    date_left, date_right = st.columns(2)
    with date_left:
        target_year = st.selectbox("Jahr", MACRO_AVAILABLE_YEARS, index=MACRO_AVAILABLE_YEARS.index(2026))
    with date_right:
        target_month = st.select_slider(
            "Monat",
            options=MACRO_AVAILABLE_MONTHS,
            value=6,
            format_func=lambda m: MONTH_NAMES[m],
        )

    target_ym = f"{target_year}-{target_month:02d}"
    vehicle_age = max(int(target_year) - int(model_year), 0)
    vehicle_age = min(vehicle_age, 30)
    st.info(f"Fahrzeugalter zum Bewertungsdatum: **{vehicle_age} Jahre** ({MONTH_NAMES[target_month]} {target_year})")

with right_column:
    st.subheader("Preisprognose")

    if model_version == "v2":
        prediction_input = build_v2_input(
            model_year=int(model_year),
            vehicle_age=vehicle_age,
            odometer=int(odometer),
            condition=float(condition),
            make=selected_make,
            model=selected_model,
            trim=trim,
            body=selected_body,
            transmission=transmission,
            state=state,
            color=color,
            interior=interior,
        )
    else:
        prediction_input = build_v1_input(
            vehicle_age=vehicle_age,
            odometer=int(odometer),
            condition=float(condition),
            make=selected_make,
            model=selected_model,
            body=selected_body,
        )

    stage1_price = float(model.predict(prediction_input)[0])
    stage2_price, cpi_multiplier = apply_stage2(stage1_price, target_ym, macro)
    final_price, seasonal_factor, seasonal_row = apply_stage3(
        stage2_price,
        selected_body,
        int(target_month),
        seasonality,
    )
    price_delta = stage2_price - stage1_price
    delta_pct = (cpi_multiplier - 1.0) * 100
    seasonal_delta = final_price - stage2_price
    seasonal_delta_pct = (seasonal_factor - 1.0) * 100

    st.metric(
        label="Finaler Preis: Markt + Saison",
        value=format_currency(final_price),
        delta=f"{seasonal_delta:+,.0f} saisonaler Effekt",
        help="Stage-2-Marktpreis x saisonaler Faktor fuer Karosserieform und Monat.",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Stage 1: Fahrzeugwert-Baseline",
        format_currency(stage1_price),
        help=(
            "Zeitneutrale V2-Vorhersage nur aus Fahrzeugmerkmalen."
            if model_version == "v2"
            else "V1-Vorhersage mit der festen Marktreferenz 2015-02."
        ),
    )
    col2.metric(
        "Stage 2: Marktpreis",
        format_currency(stage2_price),
        delta=f"{price_delta:+,.0f}",
        help="Stage-1-Basispreis x CPI-Multiplikator fuer das gewaehlte Bewertungsdatum.",
    )
    col3.metric(
        f"Saisonfaktor ({MONTH_NAMES[target_month]})",
        f"{seasonal_factor:.4f}",
        delta=f"{seasonal_delta_pct:+.1f}%",
        help="Faktor aus CPI- und fahrzeugmixbereinigten historischen Preisabweichungen.",
    )

    col4, col5 = st.columns(2)
    col4.metric(
        f"CPI-Multiplikator ({target_ym})",
        f"{cpi_multiplier:.4f}",
        delta=f"{delta_pct:+.1f}% vs. 2015",
        help="Verhaeltnis des CPI Gebrauchtwagen zum 2015-Jahresdurchschnitt (FRED: CUSR0000SETA01).",
    )
    has_recommendation = bool(seasonal_row.get("has_recommendation", False))
    best_month_value = (
        str(seasonal_row.get("best_month_name", MONTH_NAMES[target_month]))
        if has_recommendation
        else "Keine belastbare Empfehlung"
    )
    best_month_delta = (
        f"{(float(seasonal_row.get('best_factor', 1.0)) - seasonal_factor) * 100:+.1f} Prozentpunkte"
        if has_recommendation
        else None
    )
    col5.metric(
        "Bester Verkaufsmonat",
        best_month_value,
        delta=best_month_delta,
        help=(
            "Stärkster historisch beobachteter Monat. Eine Empfehlung wird nur "
            "bei mindestens zwei Monaten mit jeweils 100 Verkäufen angezeigt."
        ),
    )

    if delta_pct > 10:
        st.warning(
            f"Gebrauchtwagenpreise liegen **{delta_pct:.1f}% über** dem 2015-Niveau — "
            f"hauptsächlich durch den COVID-bedingten Angebotsengpass (2021–2022)."
        )
    elif delta_pct < -5:
        st.info(
            f"Gebrauchtwagenpreise liegen **{abs(delta_pct):.1f}% unter** dem 2015-Niveau."
        )
    else:
        st.info(f"Gebrauchtwagenpreise nahe am 2015-Referenzniveau ({delta_pct:+.1f}%).")

    seasonal_observations = int(seasonal_row.get("observations", 0))
    if seasonal_observations == 0:
        st.info(
            f"Für {MONTH_NAMES[target_month]} enthält der historische Datensatz keine Verkäufe. "
            "Die Saisonanpassung bleibt deshalb neutral."
        )
    elif seasonal_delta_pct > 2:
        st.success(
            f"Saisonal ist {MONTH_NAMES[target_month]} fuer **{selected_body}** eher stark "
            f"({seasonal_delta_pct:+.1f}%)."
        )
    elif seasonal_delta_pct < -2:
        better_month_hint = (
            f" Historisch besser: **{seasonal_row.get('best_month_name')}**."
            if has_recommendation
            else " Für einen Monatsvergleich ist die Datenbasis zu klein."
        )
        st.warning(
            f"Saisonal ist {MONTH_NAMES[target_month]} fuer **{selected_body}** eher schwach "
            f"({seasonal_delta_pct:+.1f}%).{better_month_hint}"
        )
    else:
        st.info(
            f"Saisonal liegt {MONTH_NAMES[target_month]} fuer **{selected_body}** nahe am Durchschnitt "
            f"({seasonal_delta_pct:+.1f}%)."
        )

    metric_values = metrics.get("v2_metrics", metrics.get("metrics", {}))
    mae = metric_values.get("mae")
    r2 = metric_values.get("r2")
    if mae is not None and r2 is not None:
        mq_left, mq_right = st.columns(2)
        previous_mae = metrics.get("current_model_metrics_same_test", {}).get("mae")
        if model_version == "v2" and previous_mae is not None:
            improvement_pct = (float(previous_mae) - float(mae)) / float(previous_mae) * 100
            mq_left.metric(
                "Durchschnittlicher Fehler Stage 1 V2 (MAE)",
                format_currency(float(mae)),
                delta=f"-{improvement_pct:.2f}% vs. V1 ({format_currency(float(previous_mae))})",
                delta_color="inverse",
                help="V1 und V2 wurden auf denselben 105.834 Testfahrzeugen verglichen.",
            )
        else:
            mq_left.metric("Durchschnittlicher Fehler Stage 1 (MAE)", format_currency(float(mae)))
        mq_right.metric("R² Score", f"{float(r2):.3f}")

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
        width="stretch",
        hide_index=True,
    )

st.divider()

summary_cols = st.columns(3)
summary_cols[0].metric("Trainingsdaten", f"{metrics.get('train_rows', metrics.get('rows_used', 0)):,}".replace(",", "."))
summary_cols[1].metric("Testdaten", f"{metrics.get('test_rows', 0):,}".replace(",", "."))
summary_cols[2].metric("Modell", "Stage 1 V2" if model_version == "v2" else metrics.get("model_name", "Stage 1 V1"))

with st.expander("Was passiert hier – Schritt für Schritt?"):
    st.markdown(
        f"""
**Stage 1 – Fahrzeugwert-Baseline**

1. Die App nimmt deine Fahrzeugdaten (Marke, Modell, Alter, Mileage, Zustand).
2. Das Produktionsmodell ({'V2 XGBoost-Ensemble' if model_version == 'v2' else 'V1 HistGradientBoosting'})
   berechnet daraus einen Basispreis. V2 enthält bewusst keinen Verkaufsmonat;
   Markt und Saison werden erst in Stage 2 und 3 ergänzt.

**Stage 2 – CPI-Marktpreisanpassung**

3. Der Stage-1-Basispreis wird mit dem CPI-Multiplikator für das gewählte
   Bewertungsdatum ({target_ym}) multipliziert:

   `Stage-2-Preis = Stage-1-Preis × {cpi_multiplier:.4f}`

4. Der Multiplikator kommt aus dem FRED-Datensatz *CPI Used Cars & Trucks*
   (CUSR0000SETA01), normiert auf den 2015-Jahresdurchschnitt (= 1.000).

5. Für den COVID-Angebotsengpass (2021–2022) erreichte der Multiplikator bis
   zu **1.22** (+22%). Aktuell (2026-06) liegt er stabil bei ~1.22.

**Stage 3 – Saisonale Anpassung**

6. Der Stage-2-Preis wird mit dem Faktor für Karosserieform und Zielmonat
   multipliziert: `Finaler Preis = Stage-2-Preis × {seasonal_factor:.4f}`.

7. Die Faktoren vergleichen CPI-bereinigte Verkaufspreise mit vergleichbaren
   Stage-1-Schätzungen. Monate mit wenigen Daten werden stark gedämpft; für
   Monate ohne historische Verkäufe bleibt der Faktor neutral bei 1.0.
"""
    )

with st.expander(f"Makroökonomischer Kontext – {target_ym}"):
    ctx = get_macro_context(target_ym, macro)
    ctx_rows = [
        {"Indikator": "CPI-Multiplikator (2015 = 1.000)", "Wert": f"{ctx['cpi_multiplier']:.4f}"},
    ]
    for col, label in MACRO_SIGNAL_LABELS.items():
        val = ctx.get(col)
        if val is not None:
            formatted = f"{int(val)}" if col == "recession" else f"{val:,.4g}"
            ctx_rows.append({"Indikator": label, "Wert": formatted})
    st.dataframe(pd.DataFrame(ctx_rows), width="stretch", hide_index=True)
    st.caption(
        f"Quelle: FRED (St. Louis Fed). Für Monate ohne aktuelle Daten wird der "
        f"zuletzt verfügbare Wert genutzt (Forward-Fill). Dargestellt: {ctx['year_month']}."
    )

with st.expander(f"Saisonale Datenbasis – {selected_body}"):
    body_seasonality = seasonality[seasonality["body"] == selected_body].copy()
    if not body_seasonality.empty:
        confidence_labels = {
            "high": "hoch",
            "medium": "mittel",
            "low": "niedrig",
            "no_data": "keine Daten",
        }
        body_seasonality["confidence"] = body_seasonality["confidence"].map(
            confidence_labels
        ).fillna(body_seasonality["confidence"])
        st.dataframe(
            body_seasonality[
                ["month_name", "seasonal_factor", "seasonal_delta_pct", "observations", "confidence"]
            ].rename(
                columns={
                    "month_name": "Monat",
                    "seasonal_factor": "Faktor",
                    "seasonal_delta_pct": "Effekt (%)",
                    "observations": "Verkäufe",
                    "confidence": "Datenbasis",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "August bis November sind im historischen Datensatz nicht enthalten und bleiben neutral."
        )

with st.expander("Stage-2-Backtestergebnis (historisches Testset 2014–2015)"):
    if stage2_eval:
        s1 = stage2_eval.get("stage1_metrics_historical", {})
        s2 = stage2_eval.get("stage2_metrics_historical", {})
        mult_stats = stage2_eval.get("test_multiplier_stats", {})
        cmp_data = {
            "Metrik": ["MAE ($)", "RMSE ($)", "R²", "MAPE (%)"],
            "Stage 1": [
                f"${s1.get('mae', 0):,.2f}",
                f"${s1.get('rmse', 0):,.2f}",
                f"{s1.get('r2', 0):.4f}",
                f"{s1.get('mape_percent', 0):.2f}%",
            ],
            "Stage 2": [
                f"${s2.get('mae', 0):,.2f}",
                f"${s2.get('rmse', 0):,.2f}",
                f"{s2.get('r2', 0):.4f}",
                f"{s2.get('mape_percent', 0):.2f}%",
            ],
        }
        st.dataframe(pd.DataFrame(cmp_data), width="stretch", hide_index=True)
        st.caption(
            f"CPI-Multiplikator im Testset (2014–2015): "
            f"min={mult_stats.get('min', 0):.4f} / max={mult_stats.get('max', 0):.4f} / "
            f"ø={mult_stats.get('mean', 0):.4f}. "
            f"Stage 2 verändert die historische Genauigkeit um <$1 MAE, weil die "
            f"Trainingsperiode im CPI-Basisjahr-Bereich liegt."
        )
    else:
        st.write("Noch keine Stage-2-Evaluationsdaten. Bitte `uv run python scripts/evaluate_stage2.py` ausführen.")

with st.expander("Wichtigste Einflussfaktoren (Stage 1)"):
    top_features = metrics.get("top_features", [])
    if top_features:
        st.dataframe(pd.DataFrame(top_features), width="stretch", hide_index=True)
    else:
        st.write("Noch keine Feature-Importance gespeichert.")

with st.expander("Modellgenauigkeit nach Preissegment (Stage 1)"):
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
        st.dataframe(df_seg, width="stretch", hide_index=True)
        st.caption(
            "Das Modell ist am genauesten im Mittelklasse-Segment ($10k–$20k). "
            "Budget-Fahrzeuge (MAPE ~35%) und Luxusfahrzeuge (MAPE ~21%) sind schwerer vorherzusagen."
        )
    else:
        st.write("Segmentauswertung noch nicht verfügbar.")
