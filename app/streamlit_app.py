from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
    CATBOOST_METRICS_PATH,
    build_v1_input,
    build_v2_input,
    build_catboost_input,
    get_v2_category_options,
    get_catboost_category_options,
    load_displacement_lookup,
    lookup_displacement,
    predict_stage1,
    load_production_model,
)

LOGO_PATH = PROJECT_ROOT / "app" / "assets" / "pricepilot-logo.png"
LOGO_DISPLAY_PATH = PROJECT_ROOT / "app" / "assets" / "pricepilot-logo-display.png"
STAR_COMPONENT_DIR = PROJECT_ROOT / "app" / "components" / "star_rating"
FEATURES_PATH = PROJECT_ROOT / "data" / "car_prices_features.csv"
STAGE2_EVAL_PATH = PROJECT_ROOT / "models" / "stage2_evaluation.json"
SHARED_SPLIT_BENCHMARK_PATH = (
    PROJECT_ROOT / "archive" / "models" / "artifacts" / "stage1_shared_split_model_comparison.json"
)
MACRO_PATH = PROJECT_ROOT / "data" / "macro_index.csv"
SEASONALITY_V1_PATH = (
    PROJECT_ROOT / "archive" / "models" / "artifacts" / "stage3_legacy_seasonality_factors.csv"
)
SEASONALITY_V2_PATH = PROJECT_ROOT / "models" / "stage3_seasonality_factors.csv"

MIN_SIMILAR_VEHICLES = 30
MIN_MODEL_BODY_VEHICLES = 100
LUXURY_PRICE_THRESHOLD = 40_000

MACRO_AVAILABLE_YEARS = list(range(1996, 2027))
MACRO_AVAILABLE_MONTHS = list(range(1, 13))
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez",
}
CONDITION_OPTIONS = {
    1: (1.0, "1 Stern: erhebliche Schäden und umfassender Reparaturbedarf."),
    2: (2.0, "2 Sterne: deutliche optische oder technische Mängel."),
    3: (3.0, "3 Sterne: normale altersbedingte Abnutzung und Gebrauchsspuren."),
    4: (4.0, "4 Sterne: gepflegter Zustand mit kleineren Gebrauchsspuren."),
    5: (5.0, "5 Sterne: kaum sichtbare Gebrauchsspuren und keine bekannten größeren Mängel."),
}
MILES_PER_KILOMETER = 0.621371

COLOR_LABELS = {
    "beige": "Beige", "black": "Schwarz", "blue": "Blau", "brown": "Braun",
    "burgundy": "Bordeauxrot", "charcoal": "Anthrazit", "gold": "Gold",
    "gray": "Grau", "green": "Grün", "lime": "Hellgrün", "off-white": "Cremeweiß",
    "orange": "Orange", "pink": "Rosa", "purple": "Violett", "red": "Rot",
    "silver": "Silber", "tan": "Hellbraun", "turquoise": "Türkis", "white": "Weiß",
    "yellow": "Gelb", "—": "Nicht angegeben",
}
TRANSMISSION_LABELS = {
    "automatic": "Automatik", "manual": "Schaltgetriebe", "unknown": "Nicht angegeben",
}
STATE_LABELS = {
    "ab": "Alberta (Kanada)", "al": "Alabama", "az": "Arizona", "ca": "Kalifornien",
    "co": "Colorado", "fl": "Florida", "ga": "Georgia", "hi": "Hawaii",
    "il": "Illinois", "in": "Indiana", "la": "Louisiana", "ma": "Massachusetts",
    "md": "Maryland", "mi": "Michigan", "mn": "Minnesota", "mo": "Missouri",
    "ms": "Mississippi", "nc": "North Carolina", "ne": "Nebraska",
    "nj": "New Jersey", "nm": "New Mexico", "ns": "Nova Scotia (Kanada)",
    "nv": "Nevada", "ny": "New York", "oh": "Ohio", "ok": "Oklahoma",
    "on": "Ontario (Kanada)", "or": "Oregon", "pa": "Pennsylvania",
    "pr": "Puerto Rico", "qc": "Québec (Kanada)", "sc": "South Carolina",
    "tn": "Tennessee", "tx": "Texas", "ut": "Utah", "va": "Virginia",
    "wa": "Washington", "wi": "Wisconsin",
}
BODY_LABELS = {
    "access cab": "Verlängerte Kabine", "beetle convertible": "Beetle Cabriolet",
    "cab plus": "Verlängerte Kabine (Cab Plus)", "cab plus 4": "Verlängerte Kabine (Cab Plus 4)",
    "club cab": "Verlängerte Kabine (Club Cab)", "convertible": "Cabriolet",
    "coupe": "Coupé", "crew cab": "Doppelkabine", "crewmax cab": "Große Doppelkabine",
    "cts coupe": "CTS Coupé", "cts wagon": "CTS Kombi", "cts-v coupe": "CTS-V Coupé",
    "double cab": "Doppelkabine", "e-series van": "E-Series Transporter",
    "elantra coupe": "Elantra Coupé", "extended cab": "Verlängerte Kabine",
    "g convertible": "G Cabriolet", "g coupe": "G Coupé", "g sedan": "G Limousine",
    "g37 convertible": "G37 Cabriolet", "g37 coupe": "G37 Coupé",
    "genesis coupe": "Genesis Coupé", "granturismo convertible": "GranTurismo Cabriolet",
    "hatchback": "Schrägheck", "king cab": "Verlängerte Kabine (King Cab)",
    "koup": "Koup Coupé", "mega cab": "Große Doppelkabine", "minivan": "Kleinbus",
    "promaster cargo van": "ProMaster Kastenwagen", "q60 convertible": "Q60 Cabriolet",
    "q60 coupe": "Q60 Coupé", "quad cab": "Doppelkabine (Quad Cab)",
    "regular cab": "Einzelkabine", "regular-cab": "Einzelkabine", "sedan": "Limousine",
    "supercab": "Verlängerte Kabine (SuperCab)", "supercrew": "Große Doppelkabine (SuperCrew)",
    "suv": "Geländewagen / SUV", "transit van": "Transit Transporter",
    "tsx sport wagon": "TSX Sportkombi", "van": "Transporter", "wagon": "Kombi",
    "xtracab": "Verlängerte Kabine (XtraCab)",
}
MAKE_LABELS = {
    "bmw": "BMW", "gmc": "GMC", "ram": "RAM", "mini": "MINI",
    "mercedes-benz": "Mercedes-Benz", "rolls-royce": "Rolls-Royce",
    "volkswagen": "Volkswagen",
}
# Automarken-Kürzel, die groß geschrieben bleiben müssen (z. B. CTS, AMG, RAV4).
MODEL_ACRONYMS = {
    "amg", "clk", "cls", "cr-v", "cr-z", "crx", "cts", "cts-v", "es", "gli", "glk",
    "gls", "gs", "gt", "gt-r", "gti", "gto", "gtr", "is", "ls", "mdx", "qx", "rav4",
    "rdx", "rl", "rx", "s60", "sc", "slk", "sls", "srt", "srt-8", "sti", "tl", "tsx",
    "tt", "wrx", "xc60", "xc70", "xc90", "z3", "z4", "c30", "c70",
}
# Bindewörter, die im Modellnamen klein bleiben (außer am Anfang).
MODEL_CONNECTORS = {"and", "of", "the"}


st.set_page_config(page_title="PricePilot", page_icon=str(LOGO_PATH), layout="wide")

st.markdown(
    """
    <style>
        /* --- PricePilot Branding (beide Themes) --- */
        .pp-brand {
            border-bottom: 1px solid rgba(11, 124, 255, 0.25);
            margin-bottom: 1.1rem;
            padding-bottom: 0.5rem;
        }
        .pp-kicker {
            color: #0b7cff;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.05rem;
        }
        .pp-logo { text-align: right; }
        .pp-logo img { max-width: min(100%, 380px); height: auto; object-fit: contain; }

        /* --- Preisspannen-Karte --- */
        .pp-range-card {
            border: 1px solid #cfe4ff;
            border-left: 4px solid #0b7cff;
            border-radius: 10px;
            padding: 0.9rem 1.1rem 1rem;
            background: rgba(11, 124, 255, 0.05);
            margin-bottom: 0.7rem;
        }
        .pp-range-label {
            color: #48617e;
            font-size: 0.9rem;
            font-weight: 650;
            margin-bottom: 0.2rem;
        }
        .pp-range-value {
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.15;
            color: #071d49;
        }

        /* --- Weiße Kästen mit blauer Umrandung: nur Light Mode --- */
        @media (prefers-color-scheme: light) {
            div[data-baseweb="select"] > div,
            .stNumberInput div[data-baseweb="input"] {
                background-color: #ffffff !important;
                border: 1.5px solid #0b7cff !important;
                border-radius: 8px !important;
            }
            .stNumberInput div[data-baseweb="input"]:focus-within,
            div[data-baseweb="select"] > div:focus-within {
                box-shadow: 0 0 0 2px rgba(11, 124, 255, 0.25) !important;
            }
            h1, h2, h3 { color: #071d49; }
        }
        @media (prefers-color-scheme: dark) {
            .pp-range-card { background: rgba(11, 124, 255, 0.12); }
            .pp-range-value { color: #ffffff; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def logo_data_uri() -> str:
    encoded = base64.b64encode(LOGO_DISPLAY_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_star_component = components.declare_component("pp_star_rating", path=str(STAR_COMPONENT_DIR))


def star_rating(default: float = 4.0, key: str | None = None) -> float:
    value = _star_component(default=float(default), key=key)
    if value is None:
        return float(default)
    return float(value)


@st.cache_resource
def load_model():
    return load_production_model()


@st.cache_data
def load_feature_data() -> pd.DataFrame:
    return pd.read_csv(
        FEATURES_PATH,
        usecols=[
            "make",
            "model",
            "body",
            "vehicle_age",
            "sellingprice",
            "odometer",
            "condition",
        ],
    )


@st.cache_data
def load_metrics(model_version: str) -> dict:
    if model_version == "catboost":
        path = CATBOOST_METRICS_PATH
    else:
        path = V2_METRICS_PATH if model_version == "v2" else V1_METRICS_PATH
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if model_version == "v2" and SHARED_SPLIT_BENCHMARK_PATH.exists():
        payload["shared_split_benchmark"] = json.loads(
            SHARED_SPLIT_BENCHMARK_PATH.read_text(encoding="utf-8")
        )
    return payload


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
    # CatBoost nutzt (wie V2) das zeitneutrale, reichhaltige Feature-Set → V2-Saisonfaktoren.
    path = SEASONALITY_V2_PATH if model_version in ("v2", "catboost") else SEASONALITY_V1_PATH
    return load_seasonality_factors(path)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def get_default_index(options: list, preferred) -> int:
    if preferred in options:
        return options.index(preferred)
    return 0


def title_case(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def format_make(value: str) -> str:
    return MAKE_LABELS.get(str(value), title_case(value))


def format_model(value: str) -> str:
    words = str(value).replace("_", " ").strip().split()
    formatted = []
    for index, word in enumerate(words):
        lowered = word.lower()
        if lowered in MODEL_ACRONYMS:
            formatted.append(word.upper())
        elif lowered in MODEL_CONNECTORS and index > 0:
            formatted.append(lowered)
        else:
            formatted.append(
                "-".join(
                    part.upper() if part.lower() in MODEL_ACRONYMS else part.capitalize()
                    for part in word.split("-")
                )
            )
    return " ".join(formatted)


def format_body(value: str) -> str:
    return BODY_LABELS.get(str(value), title_case(value))


def format_trim(value: str) -> str:
    if str(value) in {"—", "unknown"}:
        return "Nicht angegeben"
    return title_case(value)


def format_state(value: str) -> str:
    code = str(value).lower()
    return f"{STATE_LABELS.get(code, code.upper())} ({code.upper()})"


def format_color(value: str) -> str:
    return COLOR_LABELS.get(str(value), title_case(value))


def describe_condition(value: float) -> str:
    star = min(5, max(1, int(round(value))))
    return CONDITION_OPTIONS[star][1]


def get_price_segment(price: float) -> str:
    if price < 5_000:
        return "Budget"
    if price < 10_000:
        return "Economy"
    if price < 20_000:
        return "Mid-Range"
    if price < LUXURY_PRICE_THRESHOLD:
        return "Premium"
    return "Luxury"


def get_segment_error(metrics_payload: dict, segment: str) -> dict:
    segment_metrics = metrics_payload.get("segment_metrics", [])
    for row in segment_metrics:
        if row.get("segment") == segment:
            return row
    model_metrics = metrics_payload.get("v2_metrics", metrics_payload.get("metrics", {}))
    return {
        "segment": segment,
        "mae": model_metrics.get("mae", 0),
        "mape_percent": model_metrics.get("mape_percent", 0),
    }


def calculate_price_range(price: float, segment_error: dict) -> tuple[float, float, float]:
    mae = float(segment_error.get("mae") or 0)
    mape = float(segment_error.get("mape_percent") or 0) / 100
    uncertainty = max(mae, price * mape)
    lower_bound = max(price - uncertainty, 500)
    upper_bound = price + uncertainty
    return lower_bound, upper_bound, uncertainty


def summarize_data_basis(
    feature_data: pd.DataFrame,
    *,
    make: str,
    model: str,
    body: str,
    vehicle_age: int,
    odometer_miles: float,
    condition: float,
) -> dict:
    model_body = feature_data[
        (feature_data["make"] == make)
        & (feature_data["model"] == model)
        & (feature_data["body"] == body)
    ].copy()

    odometer_band = max(15_000, odometer_miles * 0.25)
    similar = model_body[
        (model_body["vehicle_age"].between(vehicle_age - 2, vehicle_age + 2))
        & (model_body["odometer"].between(odometer_miles - odometer_band, odometer_miles + odometer_band))
        & (model_body["condition"].between(condition - 1, condition + 1))
    ]

    return {
        "model_body_count": int(len(model_body)),
        "similar_count": int(len(similar)),
        "is_sparse": len(similar) < MIN_SIMILAR_VEHICLES or len(model_body) < MIN_MODEL_BODY_VEHICLES,
        "odometer_band": int(round(odometer_band)),
    }


data = load_feature_data()
model, model_version = load_model()
metrics = load_metrics(model_version)
stage2_eval = load_stage2_eval()
macro = load_macro()
seasonality = load_seasonality(model_version)
# Reichhaltiges Feature-Set (V2 und CatBoost) nutzt trim/state/color/… als Eingaben.
is_rich = model_version in ("v2", "catboost")
if model_version == "v2":
    v2_options = get_v2_category_options(model)
elif model_version == "catboost":
    v2_options = get_catboost_category_options()
else:
    v2_options = {}
displacement_lookup = load_displacement_lookup() if model_version == "catboost" else {}

st.markdown('<div class="pp-brand">', unsafe_allow_html=True)
header_text, header_logo = st.columns([0.6, 0.4], gap="large", vertical_alignment="center")
with header_text:
    st.markdown('<div class="pp-kicker">Used Car Pricing Intelligence</div>', unsafe_allow_html=True)
    st.title("PricePilot")
with header_logo:
    st.markdown(
        f'<div class="pp-logo"><img src="{logo_data_uri()}" alt="PricePilot Logo" /></div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

left_column, right_column = st.columns([0.95, 1.05], gap="large")

with left_column:
    st.subheader("Fahrzeugdaten")

    make_options = sorted(data["make"].dropna().unique(), key=format_make)
    selected_make = st.selectbox(
        "Marke",
        make_options,
        index=get_default_index(make_options, "bmw"),
        format_func=format_make,
    )

    make_data = data[data["make"] == selected_make]
    model_options = sorted(make_data["model"].dropna().unique(), key=format_model)
    if not model_options:
        model_options = sorted(data["model"].dropna().unique(), key=format_model)
    selected_model = st.selectbox("Modell", model_options, format_func=format_model)

    model_data = make_data[make_data["model"] == selected_model]
    body_options = sorted(model_data["body"].dropna().unique(), key=format_body)
    if not body_options:
        body_options = sorted(make_data["body"].dropna().unique(), key=format_body)
    if not body_options:
        body_options = sorted(data["body"].dropna().unique(), key=format_body)
    selected_body = st.selectbox(
        "Karosserieform",
        body_options,
        index=get_default_index(body_options, "sedan"),
        format_func=format_body,
    )

    if is_rich:
        if model_version == "catboost":
            st.caption("Das Stage-1-Modell (CatBoost) nutzt zusätzliche Fahrzeugdetails inkl. Hubraum für eine genauere Schätzung.")
        else:
            st.caption("Stage 1 V2 nutzt zusätzliche Fahrzeugdetails für eine genauere Schätzung.")
        trim_options = sorted(v2_options["trim"], key=format_trim)
        state_options = sorted(v2_options["state"], key=format_state)
        trim = st.selectbox(
            "Ausstattungsvariante", trim_options,
            index=get_default_index(trim_options, "base"),
            format_func=format_trim,
        )
        detail_left, detail_right = st.columns(2)
        with detail_left:
            transmission = st.selectbox(
                "Getriebe", v2_options["transmission"],
                index=get_default_index(v2_options["transmission"], "automatic"),
                format_func=lambda value: TRANSMISSION_LABELS.get(value, title_case(value)),
            )
            state = st.selectbox(
                "Bundesstaat / Region", state_options,
                index=get_default_index(state_options, "ca"),
                format_func=format_state,
            )
        with detail_right:
            color = st.selectbox(
                "Außenfarbe", v2_options["color"],
                index=get_default_index(v2_options["color"], "black"),
                format_func=format_color,
            )
            interior = st.selectbox(
                "Innenfarbe", v2_options["interior"],
                index=get_default_index(v2_options["interior"], "black"),
                format_func=format_color,
            )
    else:
        trim = transmission = state = color = interior = ""

    input_left, input_right = st.columns(2)
    with input_left:
        model_year = st.number_input("Baujahr", min_value=1990, max_value=2022, value=2012, step=1)
        odometer_km = st.number_input(
            "Kilometerstand (km)",
            min_value=1,
            max_value=800_000,
            value=80_000,
            step=5_000,
        )
        odometer_miles = float(odometer_km) * MILES_PER_KILOMETER
        st.caption("Für das US-Preismodell wird der Wert im Hintergrund automatisch in Meilen umgerechnet.")

    with input_right:
        st.markdown("**Fahrzeugzustand**")
        condition = star_rating(default=4.0, key="condition_stars")
        condition_label = f"{condition:g}".replace(".", ",") + " / 5 Sterne"
        st.caption(describe_condition(condition))

    if model_version == "catboost":
        suggested_disp = lookup_displacement(selected_make, selected_model, displacement_lookup)
        displacement = st.number_input(
            "Hubraum (Liter)",
            min_value=0.5,
            max_value=8.5,
            value=float(suggested_disp),
            step=0.1,
            help="Automatisch aus Marke/Modell vorgeschlagen — bei Bedarf anpassen.",
        )
    else:
        displacement = 0.0

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

    if model_version == "catboost":
        prediction_input = build_catboost_input(
            model_year=int(model_year),
            vehicle_age=vehicle_age,
            odometer=int(round(odometer_miles)),
            condition=float(condition),
            displacement=float(displacement),
            make=selected_make,
            model=selected_model,
            trim=trim,
            body=selected_body,
            transmission=transmission,
            state=state,
            color=color,
            interior=interior,
        )
    elif model_version == "v2":
        prediction_input = build_v2_input(
            model_year=int(model_year),
            vehicle_age=vehicle_age,
            odometer=int(round(odometer_miles)),
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
            odometer=int(round(odometer_miles)),
            condition=float(condition),
            make=selected_make,
            model=selected_model,
            body=selected_body,
        )

    stage1_price = predict_stage1(model, model_version, prediction_input)
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

    price_segment = get_price_segment(final_price)
    segment_error = get_segment_error(metrics, price_segment)
    data_basis = summarize_data_basis(
        data,
        make=selected_make,
        model=selected_model,
        body=selected_body,
        vehicle_age=vehicle_age,
        odometer_miles=odometer_miles,
        condition=float(condition),
    )
    lower_bound, upper_bound, _uncertainty = calculate_price_range(final_price, segment_error)
    range_text = f"{format_currency(lower_bound)} – {format_currency(upper_bound)}"

    st.markdown(
        f"""
        <div class="pp-range-card">
            <div class="pp-range-label">Geschätzte Preisspanne</div>
            <div class="pp-range-value">{range_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_recommendation = bool(seasonal_row.get("has_recommendation", False))
    best_month_number = int(seasonal_row.get("best_month", target_month))
    best_month_value = (
        MONTH_NAMES.get(best_month_number, str(best_month_number))
        if has_recommendation
        else "Keine belastbare Empfehlung"
    )
    best_month_delta = (
        f"{(float(seasonal_row.get('best_factor', 1.0)) - seasonal_factor) * 100:+.1f} Prozentpunkte"
        if has_recommendation
        else None
    )
    quick_cols = st.columns(2)
    quick_cols[0].metric(
        "Bester Verkaufsmonat",
        best_month_value,
        delta=best_month_delta,
        help=(
            "Stärkster historisch beobachteter Monat. Eine Empfehlung wird nur "
            "bei mindestens zwei Monaten mit jeweils 100 Verkäufen angezeigt."
        ),
    )
    quick_cols[1].metric(
        "Marktanpassung",
        f"{delta_pct:+.1f}%",
        help="Veränderung des Gebrauchtwagenpreisniveaus gegenüber dem 2015-Referenzniveau.",
    )

    if data_basis["is_sparse"]:
        st.warning(
            "Für diese Fahrzeugkombination liegen nur wenige vergleichbare historische Verkäufe vor "
            f"({data_basis['similar_count']} sehr ähnliche Fahrzeuge, {data_basis['model_body_count']} mit gleicher "
            "Marke, gleichem Modell und gleicher Karosserieform). Die Preisspanne ist deshalb breiter zu verstehen."
        )

    if final_price >= LUXURY_PRICE_THRESHOLD:
        st.info(
            "Hinweis zum Luxussegment: Bei sehr teuren Fahrzeugen hängt der Preis stärker von Ausstattung, "
            "Sondermodell, Unfall- und Servicehistorie sowie individuellen Merkmalen ab, die im Datensatz nur "
            "begrenzt enthalten sind. Die Preisspanne ist entsprechend vorsichtig einzuordnen."
        )

    show_price_breakdown = st.toggle(
        "Preisaufbau anzeigen",
        value=False,
        help="Zeigt die drei Rechenschritte: Fahrzeugwert, Marktpreis und Saisonfaktor.",
    )
    if show_price_breakdown:
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Stage 1: Fahrzeugwert",
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
            help="Stage-1-Basispreis × CPI-Multiplikator für das gewählte Bewertungsdatum.",
        )
        col3.metric(
            f"Stage 3: Saison ({MONTH_NAMES[target_month]})",
            f"{seasonal_factor:.4f}",
            delta=f"{seasonal_delta_pct:+.1f}%",
            help="Faktor aus CPI- und fahrzeugmixbereinigten historischen Preisabweichungen.",
        )
        st.metric(
            f"CPI-Multiplikator ({target_ym})",
            f"{cpi_multiplier:.4f}",
            delta=f"{delta_pct:+.1f}% gegenüber 2015",
            help="Verhältnis des Gebrauchtwagen-CPI zum Jahresdurchschnitt 2015 (FRED: CUSR0000SETA01).",
        )

    seasonal_observations = int(seasonal_row.get("observations", 0))
    if seasonal_observations == 0:
        st.info(
            f"Für {MONTH_NAMES[target_month]} enthält der historische Datensatz keine Verkäufe. "
            "Die Saisonanpassung bleibt deshalb neutral."
        )
    elif seasonal_delta_pct > 2:
        st.success(
            f"Saisonal ist {MONTH_NAMES[target_month]} für **{format_body(selected_body)}** eher stark "
            f"({seasonal_delta_pct:+.1f}%)."
        )
    elif seasonal_delta_pct < -2:
        better_month_hint = (
            f" Historisch besser: **{best_month_value}**."
            if has_recommendation
            else " Für einen Monatsvergleich ist die Datenbasis zu klein."
        )
        st.warning(
            f"Saisonal ist {MONTH_NAMES[target_month]} für **{format_body(selected_body)}** eher schwach "
            f"({seasonal_delta_pct:+.1f}%).{better_month_hint}"
        )
    else:
        st.info(
            f"Saisonal liegt {MONTH_NAMES[target_month]} für **{format_body(selected_body)}** nahe am Durchschnitt "
            f"({seasonal_delta_pct:+.1f}%)."
        )

    show_model_quality = st.toggle(
        "Modellgenauigkeit anzeigen",
        value=False,
        help="Zeigt die internen Testkennzahlen des Preismodells.",
    )
    if show_model_quality:
        metric_values = metrics.get("shared_split_benchmark", {}).get(
            "v2_metrics", metrics.get("v2_metrics", metrics.get("metrics", {}))
        )
        mae = metric_values.get("mae")
        r2 = metric_values.get("r2")
        if mae is not None and r2 is not None:
            mq_left, mq_right = st.columns(2)
            benchmark = metrics.get("shared_split_benchmark", {})
            previous_mae = benchmark.get("v1_metrics", {}).get(
                "mae", metrics.get("current_model_metrics_same_test", {}).get("mae")
            )
            mae = benchmark.get("v2_metrics", {}).get("mae", mae)
            if model_version == "v2" and previous_mae is not None:
                improvement_pct = (float(previous_mae) - float(mae)) / float(previous_mae) * 100
                mq_left.metric(
                    "Durchschnittlicher Fehler Stage 1 V2 (MAE)",
                    format_currency(float(mae)),
                    delta=f"-{improvement_pct:.2f}% gegenüber V1 ({format_currency(float(previous_mae))})",
                    delta_color="inverse",
                    help="V1 und V2 wurden auf denselben 105.834 Testfahrzeugen verglichen.",
                )
            else:
                mq_left.metric("Durchschnittlicher Fehler Stage 1 (MAE)", format_currency(float(mae)))
            mq_right.metric("Bestimmtheitsmaß R²", f"{float(r2):.3f}")

    display_input = {
        "Marke": format_make(selected_make),
        "Modell": format_model(selected_model),
        "Karosserieform": format_body(selected_body),
        "Baujahr": int(model_year),
        "Fahrzeugalter": vehicle_age,
        "Kilometerstand": f"{int(odometer_km):,} km".replace(",", "."),
        "Zustand": condition_label,
    }
    if model_version == "v2":
        display_input.update({
            "Ausstattungsvariante": format_trim(trim),
            "Getriebe": TRANSMISSION_LABELS.get(transmission, title_case(transmission)),
            "Bundesstaat / Region": format_state(state),
            "Außenfarbe": format_color(color),
            "Innenfarbe": format_color(interior),
        })
    st.dataframe(pd.DataFrame([display_input]), width="stretch", hide_index=True)

st.divider()

show_developer_details = st.toggle(
    "Entwicklerdetails anzeigen",
    value=False,
    help="Zeigt Modellvergleich, Backtests, Makrodaten und technische Tabellen.",
)

if show_developer_details:
    summary_cols = st.columns(3)
    summary_cols[0].metric("Trainingsdaten", f"{metrics.get('train_rows', metrics.get('rows_used', 0)):,}".replace(",", "."))
    summary_cols[1].metric("Testdaten", f"{metrics.get('test_rows', 0):,}".replace(",", "."))
    summary_cols[2].metric("Modell", "Stage 1 V2" if model_version == "v2" else metrics.get("model_name", "Stage 1 V1"))

    if model_version == "v2":
        benchmark = metrics.get("shared_split_benchmark", {})
        v2_mae = benchmark.get("v2_metrics", {}).get(
            "mae", metrics.get("v2_metrics", {}).get("mae")
        )
        v1_mae = benchmark.get("v1_metrics", {}).get(
            "mae", metrics.get("current_model_metrics_same_test", {}).get("mae")
        )
        if v1_mae is not None and v2_mae is not None:
            improvement_dollars = float(v1_mae) - float(v2_mae)
            improvement_percent = improvement_dollars / float(v1_mae) * 100
            st.subheader("Direkter Modellvergleich: V1 und V2")
            st.caption(
                "Beide Modelle wurden auf denselben 423.335 Zeilen neu trainiert und auf "
                "denselben 105.834 zuvor unangetasteten Testfahrzeugen bewertet."
            )
            comparison_cols = st.columns(3)
            comparison_cols[0].metric(
                "V1 – bisheriges Modell",
                format_currency(float(v1_mae)),
                help="Mittlerer absoluter Fehler des bisherigen Stage-1-Modells.",
            )
            comparison_cols[1].metric(
                "V2 – neues Produktionsmodell",
                format_currency(float(v2_mae)),
                delta=f"-{format_currency(improvement_dollars)} Fehler",
                delta_color="inverse",
                help="Mittlerer absoluter Fehler des neuen V2-Ensembles.",
            )
            comparison_cols[2].metric(
                "MAE-Verbesserung",
                f"{improvement_percent:.2f}%",
                delta="V2 ist genauer",
                help="Relative Verringerung des MAE von V1 auf V2.",
            )

    with st.expander("Was passiert hier – Schritt für Schritt?"):
        st.markdown(
            f"""
    **Stage 1 – Fahrzeugwert-Basiswert**

    1. Die App nimmt deine Fahrzeugdaten (Marke, Modell, Alter, Kilometerstand, Zustand).
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

    with st.expander(f"Saisonale Datenbasis – {format_body(selected_body)}"):
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
            body_seasonality["month_name"] = body_seasonality["sale_month"].map(MONTH_NAMES)
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

    with st.expander("Stage-2-Backtestergebnis (historische Testdaten 2014–2015)"):
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
                f"CPI-Multiplikator in den Testdaten (2014–2015): "
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
            st.write("Noch keine Merkmalsbedeutung gespeichert.")

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
            segment_labels = {
                "Budget": "Sehr günstig", "Economy": "Günstig", "Mid-Range": "Mittelklasse",
                "Premium": "Premium", "Luxury": "Luxusklasse",
            }
            if "Segment" in df_seg.columns:
                df_seg["Segment"] = df_seg["Segment"].map(segment_labels).fillna(df_seg["Segment"])
            st.dataframe(df_seg, width="stretch", hide_index=True)
            st.caption(
                "Das Modell ist am genauesten im Mittelklasse-Segment ($10k–$20k). "
                "Budget-Fahrzeuge (MAPE ~35%) und Luxusfahrzeuge (MAPE ~21%) sind schwerer vorherzusagen."
            )
        else:
            st.write("Segmentauswertung noch nicht verfügbar.")
