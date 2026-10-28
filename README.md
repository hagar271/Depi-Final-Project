# Customer Intelligence System

A Streamlit dashboard and an offline ML pipeline built on the **Online Retail II** dataset. It answers three business questions: where does revenue come from, which customer groups exist, and which customers are likely to stop buying in the next 90 days.

## Dashboard pages

| Page | What it shows |
|---|---|
| Overview | Revenue, orders, customers, average order value, segment and churn-risk snapshot |
| Sales Analysis | Month filter, top products, top countries |
| Customer Segmentation | RFM + K-Means segments, segment actions, customer lookup |
| Churn Prediction | Label definition, metrics, ROC / confusion matrix, model comparison, retention priority list |
| Customer Prediction | Score one customer, an uploaded transactions file, or an existing customer ID |

## Method in short

1. **Cleaning:** remove duplicates, missing Customer ID / Description, cancelled invoices, non-positive quantity or price, and non-product codes.
2. **Segmentation:** RFM per customer, `log1p` + `StandardScaler`, K-Means with k chosen by silhouette score, rule-based segment names (`segment_rules.py`).
3. **Churn:** label = no purchase in the next 90 days after a cutoff. Features use only data before the cutoff. Four training snapshots, one later test cutoff, grouped cross-validation, best model chosen by cross-validated F1.

Details: [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) (see Appendix A).

## Run it

```bash
pip install -r requirements.txt
python build_results.py path/to/online_retail_II.xlsx   # builds the result files (Windows: build_results.bat)
streamlit run app.py                                    # Windows: run_app.bat
```

Without the generated result files the app opens a **Dashboard Preview**. Its values are illustrative, not real results.

## Generated result files

Core: `rfm_kmeans_customers.csv`, `churn_model_pipeline.joblib`, `churn_model_meta.json`, `churn_scores_current_customers.csv`, `sales_kpis.json`, `sales_monthly.csv`, `sales_products.csv`, `sales_countries.csv`.

Optional: `kmeans_model.joblib`, `kmeans_scaler.joblib`, `segment_names.json`, `churn_model_comparison.csv`, `churn_test_predictions.csv`, `churn_label_summary.csv`, `churn_feature_importance.csv`.

Do **not** commit `online_retail_II.xlsx` or `clean_online_retail.csv`. Check customer-level CSV files for identifiers before committing them to a public repository.

## Tests

```bash
pip install pytest
python -m pytest tests -q     # runs on a small synthetic dataset, no real data needed
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Dashboard entry point and preview fallback |
| `demo_dashboard.py` | Preview shown when result files are absent |
| `page_*.py` | The five dashboard pages |
| `pipeline.py` | Cleaning, RFM, K-Means, churn labels, training, evaluation |
| `segment_rules.py` | The one rule that names K-Means clusters |
| `build_results.py` / `.bat` | Builds the result files from the raw dataset |
| `config.py`, `data_loader.py`, `aggregates.py`, `segments.py`, `predict.py`, `charts.py`, `utils.py` | Supporting modules |
| `tests/` | Smoke tests on synthetic data |
