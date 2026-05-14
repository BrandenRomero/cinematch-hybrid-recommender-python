from __future__ import annotations

from cinematch.utils import clamp


class HybridRecommender:
    """Weighted blend of multiple recommender models."""

    def __init__(self, models: list[object], weights: list[float] | None = None, name: str = "Hybrid re-ranker") -> None:
        if not models:
            raise ValueError("HybridRecommender needs at least one fitted model.")
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        if len(self.models) != len(self.weights):
            raise ValueError("Number of models and weights must match.")
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        self.name = name

    def predict(self, user: int, item: int) -> float:
        score = 0.0
        for model, weight in zip(self.models, self.weights):
            score += weight * float(model.predict(user, item))
        return clamp(score)

    def score(self, user: int, item: int) -> float:
        return self.predict(user, item)
