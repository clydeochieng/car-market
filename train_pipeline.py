"""Train models and export artifacts for the used-car dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "final_indian_used_car_market_dataset.csv"
CONFIG_PATH = ROOT / "project_config.json"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "price_model.joblib"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
ENCODER_META_PATH = OUTPUT_DIR / "feature_metadata.json"
DASHBOARD_JSON_PATH = ROOT / "dashboard_data.json"

# Avoid target leakage from price-derived fields
LEAKAGE_COLS = {
    "listed_price",
    "estimated_final_price",
    "depreciation_amount",
    "resale_value_percent",
    "bargain_percent",
    "days_to_sell",
}
TARGET = "estimated_final_price"
CATEGORICAL = ["brand", "model", "fuel_type", "transmission", "owner_type", "city"]
NUMERIC = [
    "year",
    "km_driven",
    "mileage",
    "engine_cc",
    "condition_rating",
    "original_price",
    "car_age",
    "demand_score",
    "brand_popularity_score",
    "price_per_km",
    "premium_car",
]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in LEAKAGE_COLS]
    X = df[[c for c in feature_cols if c != TARGET]]
    y = df[TARGET]
    return X, y


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    X, y = build_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = make_preprocessor()
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        ),
    }

    metrics: dict[str, dict] = {}
    best_name = ""
    best_rmse = float("inf")
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, estimator in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        scores = evaluate(y_test.values, preds)
        metrics[name] = scores
        fitted_pipelines[name] = pipe
        if scores["rmse"] < best_rmse:
            best_rmse = scores["rmse"]
            best_name = name

    best_pipe = fitted_pipelines[best_name]
    joblib.dump(best_pipe, MODEL_PATH)

    xgb_pipe = fitted_pipelines["XGBoost"]
    prep = xgb_pipe.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    importances = xgb_pipe.named_steps["model"].feature_importances_
    top_features = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )[:15]

    meta = {
        "target": TARGET,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "best_model": best_name,
        "top_xgb_features": [
            {"feature": str(f), "importance": float(v)} for f, v in top_features
        ],
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_metrics": metrics,
                "best_model": best_name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            },
            f,
            indent=2,
        )
    with open(ENCODER_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        project = json.load(f)

    chart_images = sorted(
        p.name for p in OUTPUT_DIR.glob("*.png")
    )
    metrics_payload = {
        "test_metrics": metrics,
        "best_model": best_name,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    dashboard_payload = {
        "project": project,
        "dataset_summary": {
            "rows": len(df),
            "brands": int(df["brand"].nunique()),
            "cities": int(df["city"].nunique()),
            "avg_estimated_final_price": float(df[TARGET].mean()),
            "median_days_to_sell": float(df["days_to_sell"].median()),
        },
        "model_metrics": metrics_payload,
        "feature_metadata": meta,
        "chart_images": chart_images,
    }
    with open(DASHBOARD_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dashboard_payload, f, indent=2)

    print(f"Best model: {best_name} (RMSE {best_rmse:,.0f})")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved dashboard bundle -> {DASHBOARD_JSON_PATH}")


if __name__ == "__main__":
    main()
