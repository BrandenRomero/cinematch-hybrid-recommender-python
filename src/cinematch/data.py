from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class Dataset:
    """Container for MovieLens-style data after id remapping."""

    users: pd.DataFrame
    items: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    user_id_to_index: dict[int, int]
    item_id_to_index: dict[int, int]
    title_to_index: dict[str, int]

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def genre_matrix(self) -> np.ndarray:
        return self.items[GENRES].to_numpy(dtype=float)

    def item_title(self, item: int) -> str:
        return str(self.items.loc[item, "title"])

    def item_genres(self, item: int) -> list[str]:
        row = self.items.loc[item]
        genres = [genre for genre in GENRES if int(row[genre]) == 1]
        return genres or ["unknown"]

    def clone_with_split(self, train: pd.DataFrame, test: pd.DataFrame) -> "Dataset":
        return Dataset(
            users=self.users,
            items=self.items,
            train=train.reset_index(drop=True),
            test=test.reset_index(drop=True),
            user_id_to_index=self.user_id_to_index,
            item_id_to_index=self.item_id_to_index,
            title_to_index=self.title_to_index,
        )


def _read_items(item_file: Path) -> pd.DataFrame:
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    items = pd.read_csv(
        item_file,
        sep="|",
        names=cols,
        encoding="latin-1",
        engine="python",
    )
    items["item_index"] = np.arange(len(items), dtype=int)
    for genre in GENRES:
        items[genre] = items[genre].fillna(0).astype(int)
    items = items.set_index("item_index", drop=False)
    return items


def _read_ratings(path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["item_id"] = ratings["item_id"].astype(int)
    return ratings


def _random_user_split(ratings: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in ratings.groupby("user_id"):
        group = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        if len(group) <= 2:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(len(group) * test_fraction)))
        n_test = min(n_test, len(group) - 1)
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[:0].copy()
    return train, test


def _map_ratings(
    ratings: pd.DataFrame,
    user_id_to_index: dict[int, int],
    item_id_to_index: dict[int, int],
) -> pd.DataFrame:
    ratings = ratings[ratings["item_id"].isin(item_id_to_index)].copy()
    ratings["user"] = ratings["user_id"].map(user_id_to_index).astype(int)
    ratings["item"] = ratings["item_id"].map(item_id_to_index).astype(int)
    return ratings[["user", "item", "rating", "timestamp", "user_id", "item_id"]].reset_index(drop=True)


def load_dataset(data_dir: str | Path, split: str = "u1", seed: int = 42, test_fraction: float = 0.2) -> Dataset:
    """Load a MovieLens 100K compatible directory.

    Expected files:
    - u.item for item metadata
    - u.data for all ratings, or u1.base/u1.test for a predefined split
    """
    data_dir = Path(data_dir)
    item_file = data_dir / "u.item"
    all_file = data_dir / "u.data"
    if not item_file.exists():
        raise FileNotFoundError(f"Missing item metadata file: {item_file}")
    if not all_file.exists() and not (data_dir / f"{split}.base").exists():
        raise FileNotFoundError(f"Missing ratings file in {data_dir}")

    items = _read_items(item_file)
    item_id_to_index = {int(row.item_id): int(row.item_index) for row in items.itertuples(index=False)}

    base_file = data_dir / f"{split}.base"
    test_file = data_dir / f"{split}.test"
    if base_file.exists() and test_file.exists():
        raw_train = _read_ratings(base_file)
        raw_test = _read_ratings(test_file)
    else:
        raw_all = _read_ratings(all_file)
        raw_train, raw_test = _random_user_split(raw_all, test_fraction=test_fraction, seed=seed)

    all_users = sorted(set(raw_train["user_id"]).union(set(raw_test["user_id"])))
    user_id_to_index = {int(user_id): idx for idx, user_id in enumerate(all_users)}
    users = pd.DataFrame({"user_index": range(len(all_users)), "user_id": all_users}).set_index("user_index")

    train = _map_ratings(raw_train, user_id_to_index, item_id_to_index)
    test = _map_ratings(raw_test, user_id_to_index, item_id_to_index)
    title_to_index = {str(row.title).lower(): int(row.item_index) for row in items.itertuples(index=False)}
    return Dataset(users, items, train, test, user_id_to_index, item_id_to_index, title_to_index)


def make_user_rated_map(ratings: pd.DataFrame) -> dict[int, dict[int, float]]:
    result: dict[int, dict[int, float]] = {}
    for row in ratings.itertuples(index=False):
        result.setdefault(int(row.user), {})[int(row.item)] = float(row.rating)
    return result


def make_user_item_sets(ratings: pd.DataFrame) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for row in ratings.itertuples(index=False):
        result.setdefault(int(row.user), set()).add(int(row.item))
    return result


def summarize_dataset(dataset: Dataset) -> dict[str, int | float]:
    total_ratings = len(dataset.train) + len(dataset.test)
    density = total_ratings / max(1, dataset.n_users * dataset.n_items)
    return {
        "users": dataset.n_users,
        "items": dataset.n_items,
        "train_ratings": int(len(dataset.train)),
        "test_ratings": int(len(dataset.test)),
        "density": round(float(density), 6),
    }


def find_item_by_title(dataset: Dataset, query: str) -> int | None:
    """Find an item by exact title first, then by case-insensitive substring."""
    key = query.strip().lower()
    if key in dataset.title_to_index:
        return dataset.title_to_index[key]
    matches = [idx for title, idx in dataset.title_to_index.items() if key in title]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[0]
    return None
