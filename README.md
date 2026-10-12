# Customer Intelligence System — Streamlit Dashboard

Online Retail II · Data Cleaning → EDA → RFM → K-Means → Churn Label → Churn Model → **Dashboard**

The dashboard **never trains anything**. It only loads files saved by the notebooks and the saved
models (`.joblib`), so the numbers on screen are exactly the numbers the team produced.

## 1. Folder structure

```
customer_intelligence_app/
├── app.py                  # entry point: sidebar navigation + error handling
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # theme
├── artifacts/              # <-- PUT THE TEAM'S OUTPUT FILES HERE (see table below)
├── src/
│   ├── config.py           # file names, thresholds, colours  (change names HERE only)
│   ├── data_loader.py      # cached loading + file validation
│   ├── segments.py         # cluster -> segment name rule
│   ├── predict.py          # thin wrappers around the saved models
│   ├── charts.py           # all Plotly charts
│   └── utils.py            # CSS, formatting, Customer-ID validation
└── views/                  # one file per page
    ├── overview.py   sales.py   segmentation.py   churn.py   prediction.py
```

## 2. Connecting the team's outputs

Copy every file from the notebook's `outputs/` folder into `artifacts/` (names are unchanged).

| File in `artifacts/` | Produced by | Used in |
|---|---|---|
| `clean_online_retail.csv` | Data Cleaning | Overview, Sales Analysis |
| `rfm_kmeans_customers.csv` | RFM + K-Means | Segmentation, Overview |
| `kmeans_model.joblib`, `kmeans_scaler.joblib` | K-Means | Customer Prediction (segment of a custom customer) |
| `churn_model_pipeline.joblib` | Churn model | Customer Prediction (live prediction) |
| `churn_model_meta.json` | Churn model | Feature order, risk thresholds, best model, metrics |
| `churn_scores_current_customers.csv` | Churn model | Customer Prediction, Overview, Churn page |
| `churn_model_comparison.csv` | Model evaluation | Churn Prediction (comparison table) |
| `churn_test_predictions.csv` | Model evaluation | Confusion matrix + ROC curves |
| `churn_label_summary.csv` | Churn label | Churn rate / distribution |
| `churn_feature_importance.csv` | Model evaluation | "What drives the prediction?" |
| `segment_names.json` | RFM + K-Means | Segment names used on every page (edit it by hand to rename a segment) |

Open **System check** in the sidebar to see which files were found.

## 3. What each page shows

| Page | Highlights |
|---|---|
| Overview | KPIs, revenue trend, country share, segment & risk snapshot, computed key findings |
| Sales Analysis | Month-range filter, top products (revenue / quantity), top countries (exclude UK) |
| Customer Segmentation | Cluster sizes, summary, segment cards with suggested actions, RFM distributions, customer lookup |
| Churn Prediction | Churn rate, confusion matrix, ROC / Precision-Recall / calibration, model comparison vs a Recency-only baseline, feature importance, **risk by segment**, **priority list with CSV download** |
| Customer Prediction | Existing customer or what-if customer → churn score, Low/Medium/High risk, segment, recommended action |

## 4. Integration rules (why predictions are consistent)

* **Feature order** comes from `churn_model_meta.json -> feature_columns`; the app re-orders the input to exactly that order.
* **Preprocessing lives inside the saved pipeline** (imputer / log1p / scaler). The app never re-implements it.
* **Risk levels** use the thresholds saved in the meta file: Low < 40% ≤ Medium < 70% ≤ High.
* **Same library versions:** install the same `scikit-learn` (and `xgboost`) version used to train. The notebook stores it in the meta file and the sidebar warns on a mismatch.
* If a file is missing or a Customer ID is invalid, the app shows a clear message instead of crashing.

## 5. Run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt  # first set scikit-learn==<version from the notebook>
streamlit run app.py
```
Then open http://localhost:8501.

## 6. Keeping the dashboard professional (without over-complicating it)

* Keep one idea per chart and give every chart a clear title.
* Show 4 KPI cards at the top of a page, then charts, then tables (inside expanders).
* Use the same colours for the same meaning (red = high risk, green = low risk).
* Add a one-line "so what?" caption under important charts.
* Keep filters few and obvious (month range, top-N, exclude UK).
