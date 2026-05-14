from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def clamp(value: float, lower: float = 1.0, upper: float = 5.0) -> float:
    """Clamp a numeric prediction to the MovieLens 1-5 rating range."""
    return float(max(lower, min(upper, value)))


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """Create a plain Markdown table for result files."""
    if not rows:
        return "No rows."
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def tokenize_title(title: str) -> list[str]:
    """Tokenize a movie title for simple metadata features."""
    title = re.sub(r"\(\d{4}\)", " ", title.lower())
    tokens = re.findall(r"[a-z0-9]+", title)
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "the", "to", "with", "part", "movie"
    }
    return [token for token in tokens if token not in stop_words and len(token) > 1]


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k(scores: Iterable[tuple[int, float]], k: int) -> list[int]:
    """Return item ids sorted by score descending."""
    return [item for item, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
