from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from cinematch.data import Dataset, make_user_item_sets
from cinematch.utils import top_k


def ndcg_at_k(ranked_items: list[int], positive_items: set[int], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in positive_items:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(positive_items), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def rating_metrics(model: object, ratings: pd.DataFrame) -> dict[str, float]:
    if len(ratings) == 0:
        return {"RMSE": 0.0, "MAE": 0.0}
    sq_errors: list[float] = []
    abs_errors: list[float] = []
    for row in ratings.itertuples(index=False):
        pred = float(model.predict(int(row.user), int(row.item)))
        err = float(row.rating) - pred
        sq_errors.append(err * err)
        abs_errors.append(abs(err))
    return {"RMSE": float(np.sqrt(np.mean(sq_errors))), "MAE": float(np.mean(abs_errors))}


def ranking_metrics(model: object, dataset: Dataset, k: int = 10, positive_threshold: float = 4.0) -> dict[str, float | int]:
    positives_by_user: dict[int, set[int]] = defaultdict(set)
    for row in dataset.test.itertuples(index=False):
        if float(row.rating) >= positive_threshold:
            positives_by_user[int(row.user)].add(int(row.item))
    rated_sets = make_user_item_sets(dataset.train)
    all_items = list(range(dataset.n_items))
    users_evaluated = 0
    precision = 0.0
    recall = 0.0
    ndcg = 0.0
    recommended: set[int] = set()
    for user, positives in positives_by_user.items():
        if not positives:
            continue
        seen = rated_sets.get(user, set())
        candidates = [item for item in all_items if item not in seen]
        if not candidates:
            continue
        scores = [(item, float(model.score(user, item))) for item in candidates]
        ranked = top_k(scores, min(k, len(candidates)))
        recommended.update(ranked)
        hits = len(set(ranked).intersection(positives))
        precision += hits / k
        recall += hits / len(positives)
        ndcg += ndcg_at_k(ranked, positives, k)
        users_evaluated += 1
    if users_evaluated == 0:
        return {f"P@{k}": 0.0, f"R@{k}": 0.0, f"NDCG@{k}": 0.0, "coverage": 0.0, "users_evaluated": 0}
    return {
        f"P@{k}": precision / users_evaluated,
        f"R@{k}": recall / users_evaluated,
        f"NDCG@{k}": ndcg / users_evaluated,
        "coverage": len(recommended) / max(1, dataset.n_items),
        "users_evaluated": users_evaluated,
    }


def evaluate_model(model: object, dataset: Dataset, k: int = 10) -> dict[str, float | str | int]:
    result: dict[str, float | str | int] = {"model": getattr(model, "name", model.__class__.__name__)}
    result.update(rating_metrics(model, dataset.test))
    result.update(ranking_metrics(model, dataset, k=k))
    return result


def format_metric_rows(results: list[dict[str, object]], k: int) -> list[dict[str, object]]:
    from cinematch.utils import format_number

    rows = []
    for result in results:
        rows.append({
            "Model": result.get("model", ""),
            "RMSE": format_number(result.get("RMSE")),
            "MAE": format_number(result.get("MAE")),
            f"P@{k}": format_number(result.get(f"P@{k}")),
            f"R@{k}": format_number(result.get(f"R@{k}")),
            f"NDCG@{k}": format_number(result.get(f"NDCG@{k}")),
            "Coverage": format_number(result.get("coverage")),
        })
    return rows
