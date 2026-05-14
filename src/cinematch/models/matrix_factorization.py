from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from cinematch.data import Dataset
from cinematch.utils import clamp


@dataclass
class UserState:
    vector: np.ndarray
    bias: float


class MatrixFactorizationModel:
    """Biased matrix factorization trained with stochastic gradient descent."""

    def __init__(
        self,
        factors: int = 24,
        epochs: int = 16,
        lr: float = 0.015,
        reg: float = 0.045,
        seed: int = 42,
        name: str = "Matrix factorization",
    ) -> None:
        self.factors = int(factors)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.reg = float(reg)
        self.seed = int(seed)
        self.name = name
        self.history: list[dict[str, float | int]] = []

    def fit(self, dataset: Dataset, validation_ratings: pd.DataFrame | None = None) -> "MatrixFactorizationModel":
        rng = np.random.default_rng(self.seed)
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        train = dataset.train[["user", "item", "rating"]].to_numpy(dtype=float)
        self.global_mean = float(train[:, 2].mean()) if len(train) else 3.0
        scale = 0.08 / np.sqrt(max(1, self.factors))
        self.user_bias = np.zeros(self.n_users, dtype=float)
        self.item_bias = np.zeros(self.n_items, dtype=float)
        self.user_factors = rng.normal(0.0, scale, size=(self.n_users, self.factors))
        self.item_factors = rng.normal(0.0, scale, size=(self.n_items, self.factors))
        self.history = []
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(train))
            for idx in order:
                user = int(train[idx, 0])
                item = int(train[idx, 1])
                rating = float(train[idx, 2])
                self._sgd_update(user, item, rating)
            row: dict[str, float | int] = {"epoch": epoch, "train_rmse": self.rmse(dataset.train)}
            if validation_ratings is not None and len(validation_ratings):
                row["validation_rmse"] = self.rmse(validation_ratings)
            self.history.append(row)
        return self

    def _sgd_update(self, user: int, item: int, rating: float) -> None:
        prediction = self._raw_predict(user, item)
        error = rating - prediction
        old_user_vector = self.user_factors[user].copy()
        old_item_vector = self.item_factors[item].copy()
        self.user_bias[user] += self.lr * (error - self.reg * self.user_bias[user])
        self.item_bias[item] += self.lr * (error - self.reg * self.item_bias[item])
        self.user_factors[user] += self.lr * (error * old_item_vector - self.reg * old_user_vector)
        self.item_factors[item] += self.lr * (error * old_user_vector - self.reg * old_item_vector)

    def _raw_predict(self, user: int, item: int) -> float:
        user_bias = self.user_bias[user] if 0 <= user < len(self.user_bias) else 0.0
        item_bias = self.item_bias[item] if 0 <= item < len(self.item_bias) else 0.0
        interaction = 0.0
        if 0 <= user < self.user_factors.shape[0] and 0 <= item < self.item_factors.shape[0]:
            interaction = float(np.dot(self.user_factors[user], self.item_factors[item]))
        return float(self.global_mean + user_bias + item_bias + interaction)

    def predict(self, user: int, item: int) -> float:
        if not hasattr(self, "user_factors"):
            raise RuntimeError("MatrixFactorizationModel must be fitted before prediction.")
        return clamp(self._raw_predict(int(user), int(item)))

    def score(self, user: int, item: int) -> float:
        return self.predict(user, item)

    def rmse(self, ratings: pd.DataFrame) -> float:
        if len(ratings) == 0:
            return 0.0
        errors = []
        for row in ratings.itertuples(index=False):
            err = float(row.rating) - self.predict(int(row.user), int(row.item))
            errors.append(err * err)
        return float(np.sqrt(np.mean(errors)))

    def mae(self, ratings: pd.DataFrame) -> float:
        if len(ratings) == 0:
            return 0.0
        errors = []
        for row in ratings.itertuples(index=False):
            errors.append(abs(float(row.rating) - self.predict(int(row.user), int(row.item))))
        return float(np.mean(errors))

    def fit_user_vector(
        self,
        seed_ratings: list[dict[str, int | float]],
        epochs: int = 120,
        lr: float = 0.035,
        reg: float | None = None,
    ) -> UserState:
        """Fit a temporary latent vector for a new user using a few demo ratings."""
        if reg is None:
            reg = self.reg
        vector = np.zeros(self.factors, dtype=float)
        bias = 0.0
        for _ in range(epochs):
            for seed in seed_ratings:
                item = int(seed["item"])
                rating = float(seed["rating"])
                item_vector = self.item_factors[item]
                pred = self.global_mean + bias + self.item_bias[item] + float(np.dot(vector, item_vector))
                error = rating - pred
                old_vector = vector.copy()
                bias += lr * (error - reg * bias)
                vector += lr * (error * item_vector - reg * old_vector)
        return UserState(vector=vector, bias=float(bias))

    def score_with_user_state(self, user_state: UserState, item: int) -> float:
        pred = self.global_mean + user_state.bias + self.item_bias[item] + float(np.dot(user_state.vector, self.item_factors[item]))
        return clamp(pred)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "factors": self.factors,
            "epochs": self.epochs,
            "lr": self.lr,
            "reg": self.reg,
            "seed": self.seed,
            "global_mean": self.global_mean,
            "user_bias": self.user_bias.tolist(),
            "item_bias": self.item_bias.tolist(),
            "user_factors": self.user_factors.tolist(),
            "item_factors": self.item_factors.tolist(),
            "history": self.history,
        }
