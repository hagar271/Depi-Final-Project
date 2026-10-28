"""The single rule that names K-Means clusters (used by pipeline.py and segments.py).

K-Means is trained on log(1+R), log(1+F), log(1+M). Each cluster's average (in that log space) is
compared with the average of ALL customers:
  * Recency  below average  -> bought recently
  * Frequency above average -> buys often
  * Monetary above average  -> spends a lot
"""
from collections import Counter

import numpy as np
import pandas as pd

import config


def name_one(z_row) -> str:
    """z_row holds the standardised Recency / Frequency / Monetary averages of one cluster."""
    recent = z_row["Recency"] < 0
    good = int(z_row["Frequency"] > 0) + int(z_row["Monetary"] > 0)
    if recent and good == 2:
        return "Champions"
    if recent and good == 1:
        return "Potential Loyalists"
    if recent:
        return "New / Promising"
    if good >= 1:
        return "At Risk"
    return "Lost / Hibernating"


def name_clusters(rfm: pd.DataFrame) -> dict:
    """rfm needs the RFM columns plus 'Cluster'. Returns {cluster id: segment name}."""
    logs = np.log1p(rfm[config.RFM_FEATURES])
    z = (logs.groupby(rfm["Cluster"]).mean() - logs.mean()) / logs.std().replace(0, 1)
    names = {int(c): name_one(r) for c, r in z.iterrows()}
    counts = Counter(names.values())              # two clusters with the same name -> add the id
    return {c: (f"{n} (Cluster {c})" if counts[n] > 1 else n) for c, n in names.items()}
