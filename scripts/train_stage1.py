from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


INPUT_PATH = Path("car_prices_macro.csv")
MODEL_PATH = Path("models/stage1_xgboost.json")
ENCODER_PATH = Path("models/stage1_encoder.pkl")

CATEGORICAL_FEATURES = ["make", "body", "transmission"]
NUMERIC_FEATURES = ["condition", "odometer", "vehicle_age"]
TARGET = "sellingprice"

# Outlier bounds — aligned with data_cleaning.md (kept but filtered here for training)
PRICE_MIN = 500
PRICE_MAX = 150_000
ODOMETER_MAX = 500_000


def load_and_prepare(path: Path) -> pd.DataFrame:
    raw_numeric = [f for f in NUMERIC_FEATURES if f != "vehicle_age"]
    df = pd.read_csv(path, usecols=CATEGORICAL_FEATURES + raw_numeric + [TARGET, "year", "saledate"])

    df["saledate"] = pd.to_datetime(df["saledate"], utc=True)
    df["vehicle_age"] = df["saledate"].dt.year - df["year"]

    # Remove outliers that were intentionally kept in the cleaned dataset
    df = df[df[TARGET].between(PRICE_MIN, PRICE_MAX)]
    df = df[df["odometer"] <= ODOMETER_MAX]

    df = df.dropna(subset=CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET])

    return df


def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[CATEGORICAL_FEATURES] = encoder.fit_transform(X_train[CATEGORICAL_FEATURES])
    X_test[CATEGORICAL_FEATURES] = encoder.transform(X_test[CATEGORICAL_FEATURES])
    return X_train, X_test, encoder


def train(input_path: Path = INPUT_PATH) -> None:
    print("1. Lade und bereite Daten vor...")
    df = load_and_prepare(input_path)
    print(f"   -> {len(df):,} Zeilen nach Outlier-Filter.")

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   -> Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("2. Encode kategorische Features...")
    X_train, X_test, encoder = encode_categoricals(X_train, X_test)

    print("3. Trainiere XGBoost (Stage 1 — Micro Model)...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="mae",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print("\n4. Evaluierung...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mean_price = y_test.mean()
    print(f"   MAE:  ${mae:,.0f}  ({mae / mean_price * 100:.1f}% des Durchschnittspreises)")
    print(f"   RMSE: ${rmse:,.0f}")
    print(f"   Durchschnittlicher Verkaufspreis (Test): ${mean_price:,.0f}")

    print("\n5. Feature Importance (Top 10)...")
    importance = pd.Series(model.feature_importances_, index=CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    print(importance.sort_values(ascending=False).head(10).to_string())

    print("\n6. Speichere Modell und Encoder...")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(MODEL_PATH)

    import pickle
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    print(f"   -> Modell: {MODEL_PATH}")
    print(f"   -> Encoder: {ENCODER_PATH}")


if __name__ == "__main__":
    train()
