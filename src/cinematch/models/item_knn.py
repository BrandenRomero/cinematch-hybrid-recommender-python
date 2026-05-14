from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from cinematch.data import Dataset, make_user_rated_map
from cinematch.utils import clamp


class ItemKNNModel:
    """Item-based collaborative filtering using adjusted cosine similarity."""

    def __init__(self, neighbors: int = 30, min_common: int = 2, shrinkage: float = 10.0, name: str = "ItemKNN CF") -> None:
        self.neighbors = int(neighbors)
        self.min_common = int(min_common)
        self.shrinkage = float(shrinkage)
        self.name = name

    def fit(self, dataset: Dataset) -> "ItemKNNModel":
        self.n_items = dataset.n_items
        self.user_ratings = make_user_rated_map(dataset.train)
        self.item_means = np.full(dataset.n_items, 3.0, dtype=float)
        sums = np.zeros(dataset.n_items, dtype=float)
        counts = np.zeros(dataset.n_items, dtype=float)
        for row in dataset.train.itertuples(index=False):
            item = int(row.item)
            sums[item] += float(row.rating)
            counts[item] += 1
        self.global_mean = float(sums.sum() / counts.sum()) if counts.sum() else 3.0
        self.item_means[:] = self.global_mean
        mask = counts > 0
        self.item_means[mask] = sums[mask] / counts[mask]
        self.similarities: list[dict[int, float]] = [dict() for _ in range(dataset.n_items)]
        self._compute_adjusted_cosine()
        return self

    def _compute_adjusted_cosine(self) -> None:
        pair_stats: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for rated in self.user_ratings.values():
            entries = [(item, rating - self.item_means[item]) for item, rating in rated.items()]
            for a in range(len(entries)):
                item_a, centered_a = entries[a]
                for b in range(a + 1, len(entries)):
                    item_b, centered_b = entries[b]
                    if item_a < item_b:
                        key = (item_a, item_b)
                        ci, cj = centered_a, centered_b
                    else:
                        key = (item_b, item_a)
                        ci, cj = centered_b, centered_a
                    stat = pair_stats[key]
                    stat[0] += ci * cj
                    stat[1] += ci * ci
                    stat[2] += cj * cj
                    stat[3] += 1
        for (i, j), (num, den_i, den_j, common) in pair_stats.items():
            if common < self.min_common or den_i <= 0 or den_j <= 0:
                continue
            raw = num / math.sqrt(den_i * den_j)
            sim = raw * (common / (common + self.shrinkage))
            if sim != 0:
                self.similarities[i][j] = float(sim)
                self.similarities[j][i] = float(sim)

    def predict(self, user: int, item: int) -> float:
        if not hasattr(self, "similarities"):
            raise RuntimeError("ItemKNNModel must be fitted before prediction.")
        rated = self.user_ratings.get(int(user), {})
        if not rated:
            return clamp(float(self.item_means[item] if 0 <= item < self.n_items else self.global_mean))
        neighbors: list[tuple[float, int, float]] = []
        for other_item, rating in rated.items():
            sim = self.similarities[item].get(other_item, 0.0)
            if sim != 0.0:
                neighbors.append((sim, other_item, rating))
        if not neighbors:
            return clamp(float(self.item_means[item]))
        neighbors.sort(key=lambda x: abs(x[0]), reverse=True)
        numerator = 0.0
        denominator = 0.0
        for sim, other_item, rating in neighbors[: self.neighbors]:
            numerator += sim * (rating - self.item_means[other_item])
            denominator += abs(sim)
        if denominator <= 0:
            return clamp(float(self.item_means[item]))
        return clamp(float(self.item_means[item] + numerator / denominator))

    def score(self, user: int, item: int) -> float:
        return self.predict(user, item)
