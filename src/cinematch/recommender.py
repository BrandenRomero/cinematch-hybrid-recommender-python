from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np

from cinematch.data import Dataset, find_item_by_title
from cinematch.models.matrix_factorization import MatrixFactorizationModel
from cinematch.utils import cosine


def parse_seed_ratings(seed_text: str, dataset: Dataset) -> list[dict[str, int | float | str]]:
    """Parse demo input like 'Toy Story (1995):5; Fargo (1996):4'."""
    seed_ratings: list[dict[str, int | float | str]] = []
    for chunk in seed_text.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        title_part, rating_part = chunk.rsplit(":", 1)
        try:
            rating = float(rating_part.strip())
        except ValueError:
            continue
        item = find_item_by_title(dataset, title_part.strip())
        if item is None:
            continue
        seed_ratings.append({
            "item": int(item),
            "item_id": int(dataset.items.loc[item, "item_id"]),
            "title": dataset.item_title(item),
            "rating": float(max(1.0, min(5.0, rating))),
        })
    seen = set()
    deduped = []
    for seed in seed_ratings:
        if seed["item"] in seen:
            continue
        seen.add(seed["item"])
        deduped.append(seed)
    return deduped


def fallback_seed_ratings(dataset: Dataset, limit: int = 5) -> list[dict[str, int | float | str]]:
    grouped = dataset.train.groupby("item")["rating"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= 2].sort_values(["mean", "count"], ascending=False).head(limit)
    seeds = []
    for row in grouped.itertuples(index=False):
        item = int(row.item)
        seeds.append({
            "item": item,
            "item_id": int(dataset.items.loc[item, "item_id"]),
            "title": dataset.item_title(item),
            "rating": 5.0,
        })
    return seeds


def _best_seed_match(model: MatrixFactorizationModel, item: int, seed_items: list[int]) -> tuple[int | None, float]:
    if not seed_items:
        return None, 0.0
    item_vec = model.item_factors[item]
    best_item = None
    best_score = -1.0
    for seed_item in seed_items:
        sim = cosine(item_vec, model.item_factors[seed_item])
        if sim > best_score:
            best_score = sim
            best_item = seed_item
    return best_item, best_score


def recommend_for_new_user(
    model: MatrixFactorizationModel,
    dataset: Dataset,
    seed_ratings: list[dict[str, int | float | str]],
    n: int = 10,
) -> list[dict[str, object]]:
    if not seed_ratings:
        seed_ratings = fallback_seed_ratings(dataset)
    user_state = model.fit_user_vector(seed_ratings)
    seen = {int(seed["item"]) for seed in seed_ratings}
    seed_items = list(seen)
    high_seed_genres = set()
    for seed in seed_ratings:
        if float(seed["rating"]) >= 4.0:
            high_seed_genres.update(dataset.item_genres(int(seed["item"])))
    candidates = []
    for item in range(dataset.n_items):
        if item in seen:
            continue
        score = model.score_with_user_state(user_state, item)
        candidates.append((item, score))
    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)[:n]
    recommendations: list[dict[str, object]] = []
    for rank, (item, score) in enumerate(ranked, start=1):
        genres = dataset.item_genres(item)
        shared_genres = sorted(set(genres).intersection(high_seed_genres))
        best_seed, sim = _best_seed_match(model, item, seed_items)
        if best_seed is None:
            similar_title = "no seed item"
        else:
            similar_title = dataset.item_title(best_seed)
        recommendations.append({
            "rank": rank,
            "item": int(item),
            "item_id": int(dataset.items.loc[item, "item_id"]),
            "title": dataset.item_title(item),
            "predicted_rating": round(float(score), 3),
            "genres": genres,
            "explanation": {
                "shared_genres_with_liked_movies": shared_genres,
                "nearest_seed_movie": similar_title,
                "latent_similarity_to_seed": round(float(sim), 3),
            },
        })
    return recommendations
