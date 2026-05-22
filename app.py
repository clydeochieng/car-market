"""
Indian Used Car Market — interactive dashboard.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "final_indian_used_car_market_dataset.csv"
CONFIG_PATH = ROOT / "project_config.json"
MODEL_PATH = ROOT / "outputs" / "price_model.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
META_PATH = ROOT / "outputs" / "feature_metadata.json"
DASHBOARD_JSON_PATH = ROOT / "dashboard_data.json"

LEAKAGE_COLS = {
    "listed_price",
    "estimated_final_price",
    "depreciation_amount",
    "resale_value_percent",
    "bargain_percent",
    "days_to_sell",
}


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def predict_row(model, row: dict, columns: list[str]) -> float:
    X = pd.DataFrame([row])[columns]
    return float(model.predict(X)[0])


def main() -> None:
    st.set_page_config(
        page_title="Indian Used Car Market",
        page_icon="🚗",
        layout="wide",
    )
    config = load_config()
    df = load_data()
    target = config.get("target_column", "estimated_final_price")
    model = load_model()

    with st.sidebar:
        st.header("Project steps")
        st.markdown(
            """
1. `pip install -r requirements.txt`
2. Open **`indian_used_car_market_analysis.ipynb`**
3. `python train_pipeline.py` → JSON + model
4. `streamlit run app.py` (this app)
5. Optional: `python -m http.server` → **`dashboard.html`**
            """
        )
        st.divider()
        st.caption("Config: `project_config.json`")
        if DASHBOARD_JSON_PATH.exists():
            st.success("`dashboard_data.json` ready")
        else:
            st.warning("Run `train_pipeline.py` for JSON bundle")

    st.title(config["project_name"])
    st.caption(f"Dataset: `{config['dataset']}` · {len(df):,} listings")

    tab_overview, tab_charts, tab_model, tab_predict = st.tabs(
        ["Overview", "Charts", "Model metrics", "Price predictor"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Listings", f"{len(df):,}")
        c2.metric("Brands", df["brand"].nunique())
        c3.metric("Cities", df["city"].nunique())
        c4.metric("Avg price", f"₹{df[target].mean():,.0f}")
        st.dataframe(df.head(20), use_container_width=True)

    with tab_charts:
        col_a, col_b = st.columns(2)
        with col_a:
            brand = st.selectbox("Brand filter", ["All"] + sorted(df["brand"].unique()))
        with col_b:
            city = st.selectbox("City filter", ["All"] + sorted(df["city"].unique()))
        filtered = df.copy()
        if brand != "All":
            filtered = filtered[filtered["brand"] == brand]
        if city != "All":
            filtered = filtered[filtered["city"] == city]

        fig1 = px.histogram(filtered, x=target, nbins=30, title="Price distribution")
        st.plotly_chart(fig1, use_container_width=True)

        by_brand = (
            filtered.groupby("brand", as_index=False)[target]
            .mean()
            .sort_values(target, ascending=False)
            .head(12)
        )
        fig2 = px.bar(by_brand, x="brand", y=target, title="Average price by brand")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(filtered, x="fuel_type", y=target, title="Price by fuel type")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(
            filtered,
            x="km_driven",
            y=target,
            color="premium_car",
            title="Price vs km driven",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with tab_model:
        bundle = None
        if DASHBOARD_JSON_PATH.exists():
            with open(DASHBOARD_JSON_PATH, encoding="utf-8") as f:
                bundle = json.load(f)
            st.subheader("dashboard_data.json")
            st.json(bundle)
        elif METRICS_PATH.exists():
            with open(METRICS_PATH, encoding="utf-8") as f:
                st.json(json.load(f))
        else:
            st.warning(
                "No metrics yet. Run the notebook or `python train_pipeline.py` first."
            )
        meta = None
        if bundle:
            meta = bundle.get("feature_metadata")
        elif META_PATH.exists():
            with open(META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
        if meta:
            st.subheader("Top XGBoost features")
            st.dataframe(
                pd.DataFrame(meta.get("top_xgb_features", [])),
                use_container_width=True,
            )

    with tab_predict:
        if model is None:
            st.error("Model not found. Run `python train_pipeline.py` or the notebook.")
            return

        feature_cols = [c for c in df.columns if c not in LEAKAGE_COLS and c != target]
        sample = df.iloc[0]
        st.subheader("Enter car details")

        col1, col2, col3 = st.columns(3)
        with col1:
            brand_v = st.selectbox("Brand", sorted(df["brand"].unique()), index=0)
            models = sorted(df[df["brand"] == brand_v]["model"].unique())
            model_v = st.selectbox("Model", models)
            year_v = st.number_input("Year", 2010, 2026, int(sample["year"]))
            fuel_v = st.selectbox("Fuel", sorted(df["fuel_type"].unique()))
        with col2:
            trans_v = st.selectbox("Transmission", sorted(df["transmission"].unique()))
            km_v = st.number_input("Km driven", 0, 300000, int(sample["km_driven"]))
            owner_v = st.selectbox("Owner type", sorted(df["owner_type"].unique()))
            city_v = st.selectbox("City", sorted(df["city"].unique()))
        with col3:
            mileage_v = st.number_input("Mileage (km/l)", 10, 30, int(sample["mileage"]))
            engine_v = st.number_input("Engine CC", 800, 3000, int(sample["engine_cc"]))
            cond_v = st.slider("Condition rating", 1, 10, int(sample["condition_rating"]))
            orig_v = st.number_input(
                "Original price (INR)", 100000, 5000000, int(sample["original_price"])
            )

        car_age_v = 2026 - year_v
        demand_v = st.slider("Demand score", -20.0, 20.0, float(sample["demand_score"]))
        pop_v = st.slider(
            "Brand popularity", 0.0, 70.0, float(sample["brand_popularity_score"])
        )
        ppm_v = st.number_input("Price per km", 0.0, 100.0, float(sample["price_per_km"]))
        premium_v = st.selectbox("Premium car", [0, 1])

        row = {
            "brand": brand_v,
            "model": model_v,
            "year": year_v,
            "fuel_type": fuel_v,
            "transmission": trans_v,
            "km_driven": km_v,
            "owner_type": owner_v,
            "city": city_v,
            "mileage": mileage_v,
            "engine_cc": engine_v,
            "condition_rating": cond_v,
            "original_price": orig_v,
            "car_age": car_age_v,
            "demand_score": demand_v,
            "brand_popularity_score": pop_v,
            "price_per_km": ppm_v,
            "premium_car": premium_v,
        }

        if st.button("Predict price", type="primary"):
            pred = predict_row(model, row, feature_cols)
            st.success(f"Estimated final price: **₹{pred:,.0f}**")


if __name__ == "__main__":
    main()
