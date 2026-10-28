"""Gives each K-Means cluster a business-friendly segment name.

The naming rule itself lives in segment_rules.py (shared with pipeline.py).
The team can override the names with artifacts/segment_names.json.
"""
import pandas as pd

import data_loader
import segment_rules

SEGMENT_INFO = {
    "Champions": (
        "Bought recently, buy often and spend the most.",
        "Reward them: loyalty perks, early access to new products, ask for reviews."),
    "Potential Loyalists": (
        "Bought recently and are strong on either frequency or spending.",
        "Nurture them: cross-sell related products, invite them to a loyalty programme."),
    "New / Promising": (
        "Bought recently but with low frequency and low spending so far.",
        "Onboard them: welcome offers and a nudge to make the second purchase."),
    "At Risk": (
        "Used to be valuable customers but have not bought for a long time.",
        "Win them back: personal offer or a 'we miss you' message before they are lost."),
    "Lost / Hibernating": (
        "Long time since the last purchase, with low frequency and low spending.",
        "Low-cost reactivation only (e-mail / discount); do not over-invest."),
}


def base_name(segment: str) -> str:
    return segment.split(" (Cluster")[0]


def describe(segment: str):
    return SEGMENT_INFO.get(base_name(segment), ("Custom segment defined by the team.", "—"))


def build_segment_map(rfm: pd.DataFrame) -> dict:
    """cluster id -> segment name (the shared rule, then any manual overrides)."""
    names = segment_rules.name_clusters(rfm)
    names.update(data_loader.load_segment_overrides())
    return names


def cluster_summary(rfm: pd.DataFrame, segment_map: dict) -> pd.DataFrame:
    g = rfm.groupby("Cluster")
    out = pd.DataFrame({
        "Customers": g.size(),
        "Avg Recency (days)": g["Recency"].mean(),
        "Avg Frequency (orders)": g["Frequency"].mean(),
        "Avg Monetary (£)": g["Monetary"].mean(),
        "Total Monetary (£)": g["Monetary"].sum(),
    })
    out["% of customers"] = out["Customers"] / out["Customers"].sum() * 100
    out["% of revenue"] = out["Total Monetary (£)"] / out["Total Monetary (£)"].sum() * 100
    out.insert(0, "Segment", out.index.map(segment_map))
    return out.sort_values("Avg Monetary (£)", ascending=False)
