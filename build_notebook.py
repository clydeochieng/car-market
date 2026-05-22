"""Generate indian_used_car_market_analysis.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "indian_used_car_market_analysis.ipynb"

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }

cells = [
md("""# Indian Used Car Market — Full Analysis

**Goal:** Explore the used-car dataset, visualize market patterns, and build regression models (including **XGBoost**) to predict **estimated final sale price**.

**Artifacts:** Models and metrics are saved under `outputs/` for the Streamlit app (`app.py`)."""),

code("""# Core imports
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (10, 5)

ROOT = Path(".").resolve()
DATA_PATH = ROOT / "final_indian_used_car_market_dataset.csv"
CONFIG_PATH = ROOT / "project_config.json"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

print(config["project_name"])"""),

code("""df = pd.read_csv(DATA_PATH)
TARGET = config["target_column"]
print(f"Shape: {df.shape}")
df.head()"""),

md("## 1. Data quality & overview"),

code("""print(df.info())
print("\\nMissing values:\\n", df.isnull().sum())
df.describe().T"""),

code("""# Duplicate check
print("Duplicates:", df.duplicated().sum())

# Premium vs standard mix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["premium_car"].value_counts().plot(kind="bar", ax=axes[0], color=["#4C78A8", "#F58518"])
axes[0].set_title("Premium vs non-premium")
axes[0].set_xlabel("premium_car")
sns.histplot(df[TARGET], bins=30, kde=True, ax=axes[1], color="#54A24B")
axes[1].set_title("Distribution of estimated final price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_price_distribution.png", dpi=120)
plt.show()"""),

md("## 2. Exploratory charts"),

code("""# Average price by brand (top 10 by volume)
top_brands = df["brand"].value_counts().head(10).index
brand_avg = (
    df[df["brand"].isin(top_brands)]
    .groupby("brand")[TARGET]
    .mean()
    .sort_values(ascending=False)
)
plt.figure(figsize=(10, 5))
sns.barplot(x=brand_avg.index, y=brand_avg.values, hue=brand_avg.index, legend=False)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Avg estimated final price (INR)")
plt.title("Average resale price by brand (top 10)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_price_by_brand.png", dpi=120)
plt.show()"""),

code("""# City and fuel type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="city", y=TARGET, ax=axes[0])
axes[0].tick_params(axis="x", rotation=45)
axes[0].set_title("Price by city")
sns.boxplot(data=df, x="fuel_type", y=TARGET, ax=axes[1])
axes[1].set_title("Price by fuel type")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_city_fuel_boxplots.png", dpi=120)
plt.show()"""),

code("""# Correlation among numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_correlation_heatmap.png", dpi=120)
plt.show()"""),

code("""# Scatter relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.scatterplot(data=df, x="km_driven", y=TARGET, hue="premium_car", ax=axes[0, 0], alpha=0.7)
sns.scatterplot(data=df, x="car_age", y=TARGET, ax=axes[0, 1], alpha=0.7)
sns.scatterplot(data=df, x="condition_rating", y=TARGET, ax=axes[1, 0], alpha=0.7)
sns.scatterplot(data=df, x="original_price", y=TARGET, ax=axes[1, 1], alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_scatter_relationships.png", dpi=120)
plt.show()"""),

code("""# Days to sell & depreciation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["days_to_sell"], bins=20, kde=True, ax=axes[0])
axes[0].set_title("Days to sell")
sns.scatterplot(data=df, x="depreciation_amount", y="resale_value_percent", ax=axes[1], alpha=0.7)
axes[1].set_title("Depreciation vs resale %")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_time_depreciation.png", dpi=120)
plt.show()"""),

md("## 3. Machine learning — predict final price"),

code("""LEAKAGE_COLS = {
    "listed_price", "estimated_final_price", "depreciation_amount",
    "resale_value_percent", "bargain_percent", "days_to_sell",
}
CATEGORICAL = ["brand", "model", "fuel_type", "transmission", "owner_type", "city"]
NUMERIC = [
    "year", "km_driven", "mileage", "engine_cc", "condition_rating",
    "original_price", "car_age", "demand_score", "brand_popularity_score",
    "price_per_km", "premium_car",
]

feature_cols = [c for c in df.columns if c not in LEAKAGE_COLS and c != TARGET]
X = df[feature_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])

def metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

print(f"Train: {len(X_train)} | Test: {len(X_test)}")"""),

code("""models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    ),
}

results = []
fitted = {}
for name, est in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", est)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    m = metrics(y_test, pred)
    m["Model"] = name
    results.append(m)
    fitted[name] = pipe
    cv = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")
    print(f"{name}: RMSE={m['RMSE']:,.0f} | R2={m['R2']:.3f} | CV RMSE={-cv.mean():,.0f}")

results_df = pd.DataFrame(results).set_index("Model")
results_df"""),

code("""# Model comparison chart
plot_df = results_df.reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=plot_df, x="Model", y="RMSE", ax=axes[0])
axes[0].set_title("Test RMSE (lower is better)")
sns.barplot(data=plot_df, x="Model", y="R2", ax=axes[1])
axes[1].set_ylim(0, 1)
axes[1].set_title("Test R² (higher is better)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_model_comparison.png", dpi=120)
plt.show()"""),

code("""# XGBoost feature importance
import joblib

xgb_pipe = fitted["XGBoost"]
feat_names = xgb_pipe.named_steps["prep"].get_feature_names_out()
imp = xgb_pipe.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values(
    "importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, y="feature", x="importance", hue="feature", legend=False)
plt.title("Top 15 XGBoost feature importances")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_xgb_feature_importance.png", dpi=120)
plt.show()

imp_df"""),

code("""# Actual vs predicted (XGBoost)
pred_xgb = xgb_pipe.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_test, pred_xgb, alpha=0.7)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[0].set_title("XGBoost: Actual vs Predicted")
residuals = y_test - pred_xgb
sns.histplot(residuals, kde=True, ax=axes[1])
axes[1].set_title("Residual distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_xgb_diagnostics.png", dpi=120)
plt.show()"""),

code("""# Save best model & metrics JSON
best_model_name = results_df["RMSE"].idxmin()
best_pipe = fitted[best_model_name]

joblib.dump(best_pipe, OUTPUT_DIR / "price_model.joblib")

metrics_out = {
    "test_metrics": results_df.reset_index().to_dict(orient="records"),
    "best_model": best_model_name,
}
with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2)

print(f"Best model: {best_model_name}")
print("Saved outputs/price_model.joblib and outputs/model_metrics.json")"""),

md("""## 4. Key takeaways

- **Target:** `estimated_final_price` — fair resale value after negotiation.
- **XGBoost** typically captures non-linear effects (brand, city, mileage, condition) better than linear baselines on tabular data.
- **Watch leakage:** do not use `listed_price`, `bargain_percent`, or post-sale fields when training.
- **Next step:** run `streamlit run app.py` for an interactive dashboard and live price estimates."""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH}")
