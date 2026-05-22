# Indian Used Car Market — Pricing & Demand Intelligence

End-to-end project: notebook analysis, **JSON** artifacts, **Streamlit** app, and **HTML** report.

---

## Step-by-step

### Step 1 — Install dependencies

```bash
cd archive
pip install -r requirements.txt
```

### Step 2 — Run the notebook (EDA + charts + XGBoost)

```bash
jupyter notebook indian_used_car_market_analysis.ipynb
```

Run all cells. Charts save under `outputs/*.png`.

### Step 3 — Export JSON + model (for app & HTML)

```bash
python train_pipeline.py
```

Creates:

| File | Purpose |
|------|---------|
| `project_config.json` | Project metadata (always present) |
| `dashboard_data.json` | **Single bundle** for HTML + app |
| `outputs/model_metrics.json` | Per-model scores |
| `outputs/feature_metadata.json` | XGBoost feature importances |
| `outputs/price_model.joblib` | Best model for predictions |

### Step 4 — Interactive app (`app.py`)

```bash
streamlit run app.py
```

Tabs: overview, Plotly charts, model metrics, **price predictor**.

### Step 5 — Static HTML report (`dashboard.html`)

Browsers block local JSON unless you use a tiny server:

```bash
python -m http.server 8000
```

Open: [http://localhost:8000/dashboard.html](http://localhost:8000/dashboard.html)

---

## All deliverables

| File | Type |
|------|------|
| `indian_used_car_market_analysis.ipynb` | Analysis |
| `project_config.json` | Config JSON |
| `dashboard_data.json` | Results JSON (after Step 3) |
| `app.py` | Streamlit dashboard |
| `dashboard.html` | Static HTML report |
| `train_pipeline.py` | Train + write JSON/model |
| `final_indian_used_car_market_dataset.csv` | Data |

---

## Notes

- Price-leaking columns (`listed_price`, `bargain_percent`, etc.) are excluded from training.
- ~200 rows: metrics depend on the train/test split; the notebook uses cross-validation for extra stability.
