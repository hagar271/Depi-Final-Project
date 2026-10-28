"""Smoke tests on a small SYNTHETIC dataset (the real Online Retail II file is not needed).

Run:  python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import pipeline  # noqa: E402
import segment_rules  # noqa: E402

ALLOWED_SEGMENTS = {"Champions", "Potential Loyalists", "New / Promising", "At Risk", "Lost / Hibernating"}


def make_raw(n_customers=300, days=730, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2010-01-01")
    rows, inv = [], 10000
    for cid in range(12000, 12000 + n_customers):
        active_until = rng.integers(60, days)                    # some customers stop buying early
        rate = rng.choice([0.01, 0.03, 0.08])
        for d in range(0, active_until):
            if rng.random() < rate:
                inv += 1
                for _ in range(rng.integers(1, 5)):
                    rows.append((str(inv), f"P{rng.integers(1, 60)}", "ITEM", int(rng.integers(1, 12)),
                                 start + pd.Timedelta(days=d, hours=int(rng.integers(8, 18))),
                                 float(rng.uniform(1, 20)), float(cid), "United Kingdom"))
    raw = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
                                      "Price", "Customer ID", "Country"])
    dirty = raw.sample(40, random_state=1).copy()                # rows the cleaning step must remove
    dirty["Invoice"] = "C" + dirty["Invoice"]
    dirty["Quantity"] = -dirty["Quantity"]
    no_cust = raw.sample(40, random_state=2).copy()
    no_cust["Customer ID"] = np.nan
    postage = raw.sample(20, random_state=3).copy()
    postage["StockCode"] = "POST"
    return pd.concat([raw, dirty, no_cust, postage, raw.head(25)], ignore_index=True)   # last part = duplicates


@pytest.fixture(scope="module")
def raw():
    return make_raw()


def test_cleaning_removes_bad_rows(raw):
    df = pipeline.clean_transactions(raw)
    assert not df["Invoice"].astype(str).str.startswith("C").any()
    assert df["Customer ID"].notna().all()
    assert (df["Quantity"] > 0).all() and (df["Price"] > 0).all()
    assert not (df["StockCode"] == "POST").any()
    assert not df.duplicated().any()


def test_features_use_only_data_before_cutoff(raw):
    df = pipeline.clean_transactions(raw)
    cutoff = df["InvoiceDate"].min() + pd.Timedelta(days=400)
    full = pipeline.build_features(df, cutoff)
    truncated = pipeline.build_features(df[df["InvoiceDate"] < cutoff], cutoff)   # future rows removed
    pd.testing.assert_frame_equal(full.sort_index(), truncated.sort_index())


def test_segment_rule_names():
    z = pd.DataFrame({"Recency": [-1, -1, -1, 1, 1], "Frequency": [1, 1, -1, 1, -1],
                      "Monetary": [1, -1, -1, -1, -1]})
    assert [segment_rule for segment_rule in z.apply(segment_rules.name_one, axis=1)] == [
        "Champions", "Potential Loyalists", "New / Promising", "At Risk", "Lost / Hibernating"]


def test_full_pipeline_writes_artifacts(raw, tmp_path):
    summary = pipeline.run_pipeline(raw, tmp_path, fast=True, use_xgb=False)
    for key in ("rfm", "churn_model", "churn_meta", "churn_scores", "kmeans_model", "segment_names"):
        assert (tmp_path / config.ARTIFACTS[key][0]).exists(), key
    meta = json.loads((tmp_path / config.ARTIFACTS["churn_meta"][0]).read_text())
    assert meta["best_model"] in meta["cv_f1"]                       # winner chosen by cross-validation
    assert meta["target"].endswith("90 days)")
    rfm = pd.read_csv(tmp_path / config.ARTIFACTS["rfm"][0])
    assert {n.split(" (Cluster")[0] for n in rfm["Segment"].unique()} <= ALLOWED_SEGMENTS
    assert 0 <= summary["test_metrics"]["ROC-AUC"] <= 1
