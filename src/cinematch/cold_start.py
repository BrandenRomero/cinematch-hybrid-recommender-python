from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cinematch.data import Dataset, GENRES
from cinematch.evaluation import ndcg_at_k
from cinematch.utils import clamp, tokenize_title, top_k


@dataclass
class ColdStartSplit:
    warm_dataset: Dataset
    cold_items: set[int]
    cold_ratings: pd.DataFrame


def select_cold_items(dataset: Dataset, count: int = 160, min_ratings: int = 15, seed: int = 142) -> set[int]:
    all_ratings = pd.concat([dataset.train, dataset.test], ignore_index=True)
    counts = all_ratings.groupby("item").size()
    eligible = [int(item) for item, n in counts.items() if int(n) >= min_ratings]
    if not eligible:
        return set()
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    return set(eligible[: min(count, len(eligible))])


def create_cold_start_split(dataset: Dataset, count: int = 160, min_ratings: int = 15, seed: int = 142) -> ColdStartSplit:
    cold_items = select_cold_items(dataset, count=count, min_ratings=min_ratings, seed=seed)
    if not cold_items:
        empty = dataset.train.iloc[:0].copy()
        return ColdStartSplit(dataset.clone_with_split(dataset.train, dataset.test), set(), empty)
    warm_train = dataset.train[~dataset.train["item"].isin(cold_items)].copy()
    warm_test = dataset.test[~dataset.test["item"].isin(cold_items)].copy()
    cold_ratings = pd.concat([dataset.train, dataset.test], ignore_index=True)
    cold_ratings = cold_ratings[cold_ratings["item"].isin(cold_items)].copy().reset_index(drop=True)
    return ColdStartSplit(dataset.clone_with_split(warm_train, warm_test), cold_items, cold_ratings)


def build_title_vocabulary(dataset: Dataset, item_indices: set[int], vocab_limit: int = 120) -> list[str]:
    counts: Counter[str] = Counter()
    for item in item_indices:
        counts.update(tokenize_title(dataset.item_title(item)))
    return [token for token, _ in counts.most_common(vocab_limit)]


def feature_matrix(dataset: Dataset, item_indices: list[int], vocabulary: list[str]) -> np.ndarray:
    token_to_col = {token: idx for idx, token in enumerate(vocabulary)}
    X = np.zeros((len(item_indices), 1 + len(GENRES) + len(vocabulary)), dtype=float)
    for row_idx, item in enumerate(item_indices):
        X[row_idx, 0] = 1.0
        X[row_idx, 1 : 1 + len(GENRES)] = dataset.genre_matrix[item]
        for token in tokenize_title(dataset.item_title(item)):
            col = token_to_col.get(token)
            if col is not None:
                X[row_idx, 1 + len(GENRES) + col] = 1.0
    return X


class GenreProfileColdStart:
    """Cold-item baseline that matches user genre profiles to item genres."""

    def __init__(self, name: str = "Cold genre profile") -> None:
        self.name = name

    def fit(self, dataset: Dataset) -> "GenreProfileColdStart":
        self.genre_matrix = dataset.genre_matrix
        self.global_mean = float(dataset.train["rating"].mean()) if len(dataset.train) else 3.0
        self.user_profiles = np.zeros((dataset.n_users, self.genre_matrix.shape[1]), dtype=float)
        for row in dataset.train.itertuples(index=False):
            weight = float(row.rating) - self.global_mean
            self.user_profiles[int(row.user)] += weight * self.genre_matrix[int(row.item)]
        return self

    def score(self, user: int, item: int, dataset: Dataset | None = None) -> float:
        profile = self.user_profiles[user]
        item_vec = self.genre_matrix[item]
        denom = np.linalg.norm(profile) * np.linalg.norm(item_vec)
        if denom <= 1e-12:
            return self.global_mean
        return clamp(self.global_mean + 1.25 * float(np.dot(profile, item_vec) / denom))

    def predict(self, user: int, item: int) -> float:
        return self.score(user, item)


class MetadataLatentMapper:
    """Ridge regression mapper from item metadata to matrix-factorization item factors."""

    def __init__(self, ridge_lambda: float = 2.0, vocab_limit: int = 120, name: str = "Metadata latent mapper") -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.vocab_limit = int(vocab_limit)
        self.name = name

    def fit(self, dataset: Dataset, matrix_factorization_model: object, warm_items: set[int] | None = None) -> "MetadataLatentMapper":
        if warm_items is None:
            warm_items = set(dataset.train["item"].unique().astype(int))
        self.dataset = dataset
        self.mf = matrix_factorization_model
        self.vocabulary = build_title_vocabulary(dataset, warm_items, self.vocab_limit)
        warm_items_list = sorted(warm_items)
        X = feature_matrix(dataset, warm_items_list, self.vocabulary)
        Y = matrix_factorization_model.item_factors[warm_items_list]
        ridge = self.ridge_lambda * np.eye(X.shape[1], dtype=float)
        ridge[0, 0] = 0.0
        self.factor_weights = np.linalg.solve(X.T @ X + ridge, X.T @ Y)
        bias_targets = matrix_factorization_model.item_bias[warm_items_list]
        self.bias_weights = np.linalg.solve(X.T @ X + ridge, X.T @ bias_targets)
        return self

    def predict_item_vector(self, item: int) -> np.ndarray:
        X = feature_matrix(self.dataset, [item], self.vocabulary)
        return X @ self.factor_weights

    def predict_item_bias(self, item: int) -> float:
        X = feature_matrix(self.dataset, [item], self.vocabulary)
        return float(X @ self.bias_weights)

    def score(self, user: int, item: int, dataset: Dataset | None = None) -> float:
        item_vector = self.predict_item_vector(item).reshape(-1)
        item_bias = self.predict_item_bias(item)
        pred = self.mf.global_mean + self.mf.user_bias[user] + item_bias + float(np.dot(self.mf.user_factors[user], item_vector))
        return clamp(pred)

    def predict(self, user: int, item: int) -> float:
        return self.score(user, item)


def evaluate_cold_ranking(
    model: object,
    dataset: Dataset,
    cold_items: set[int],
    cold_ratings: pd.DataFrame,
    k: int = 10,
    positive_threshold: float = 4.0,
) -> dict[str, float | int | str]:
    positives_by_user: dict[int, set[int]] = defaultdict(set)
    for row in cold_ratings.itertuples(index=False):
        if float(row.rating) >= positive_threshold:
            positives_by_user[int(row.user)].add(int(row.item))
    if not cold_items:
        return {"model": getattr(model, "name", model.__class__.__name__), f"P@{k}": 0.0, f"R@{k}": 0.0, f"NDCG@{k}": 0.0, "coverage": 0.0, "users_evaluated": 0}
    cold_item_list = sorted(cold_items)
    users_evaluated = 0
    precision = 0.0
    recall = 0.0
    ndcg = 0.0
    recommended: set[int] = set()
    for user, positives in positives_by_user.items():
        if not positives:
            continue
        scores = [(item, float(model.score(user, item, dataset))) for item in cold_item_list]
        ranked = top_k(scores, min(k, len(cold_item_list)))
        recommended.update(ranked)
        hits = len(set(ranked).intersection(positives))
        precision += hits / k
        recall += hits / len(positives)
        ndcg += ndcg_at_k(ranked, positives, k)
        users_evaluated += 1
    if users_evaluated == 0:
        precision = recall = ndcg = 0.0
    else:
        precision /= users_evaluated
        recall /= users_evaluated
        ndcg /= users_evaluated
    return {
        "model": getattr(model, "name", model.__class__.__name__),
        f"P@{k}": precision,
        f"R@{k}": recall,
        f"NDCG@{k}": ndcg,
        "coverage": len(recommended) / max(1, len(cold_items)),
        "users_evaluated": users_evaluated,
    }
