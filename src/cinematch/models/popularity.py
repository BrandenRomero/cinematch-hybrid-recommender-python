from __future__ import annotations

import numpy as np

from cinematch.data import Dataset
from cinematch.utils import clamp


class PopularityModel:
    """Bayesian-smoothed item popularity baseline."""

    def __init__(self, m: float = 25.0, name: str = "Popularity") -> None:
        self.m = float(m)
        self.name = name

    def fit(self, dataset: Dataset) -> "PopularityModel":
        sums = np.zeros(dataset.n_items, dtype=float)
        counts = np.zeros(dataset.n_items, dtype=float)
        for row in dataset.train.itertuples(index=False):
            sums[int(row.item)] += float(row.rating)
            counts[int(row.item)] += 1.0
        self.global_mean = float(sums.sum() / counts.sum()) if counts.sum() > 0 else 3.0
        self.item_counts = counts
        self.item_scores = (sums + self.m * self.global_mean) / np.maximum(counts + self.m, 1e-12)
        return self

    def predict(self, user: int, item: int) -> float:
        if not hasattr(self, "item_scores"):
            raise RuntimeError("PopularityModel must be fitted before prediction.")
        if item < 0 or item >= len(self.item_scores):
            return self.global_mean
        return clamp(float(self.item_scores[item]))

    def score(self, user: int, item: int) -> float:
        return self.predict(user, item)
