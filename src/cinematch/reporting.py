from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from cinematch.utils import ensure_dir, format_number, markdown_table


def results_markdown(rows: list[dict[str, Any]], columns: list[str], title: str) -> str:
    return f"# {title}\n\n" + markdown_table(rows, columns) + "\n"


def render_demo_html(seed_ratings: list[dict[str, Any]], recommendations: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    seed_items = "".join(
        f"<li><strong>{html.escape(str(seed['title']))}</strong> - rating {format_number(float(seed['rating']), 1)}</li>"
        for seed in seed_ratings
    )
    rec_items = []
    for rec in recommendations:
        explanation = rec.get("explanation", {})
        shared = ", ".join(explanation.get("shared_genres_with_liked_movies", [])) or "no direct genre overlap"
        rec_items.append(
            "<li>"
            f"<h3>#{rec['rank']} {html.escape(str(rec['title']))} "
            f"<span>{format_number(float(rec['predicted_rating']), 2)} stars</span></h3>"
            f"<p><strong>Genres:</strong> {html.escape(', '.join(rec.get('genres', [])))}</p>"
            f"<p><strong>Why:</strong> shared genres: {html.escape(shared)}; "
            f"nearest seed movie: {html.escape(str(explanation.get('nearest_seed_movie', '')))} "
            f"(latent similarity {explanation.get('latent_similarity_to_seed', 0)}).</p>"
            "</li>"
        )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CineMatch Demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.5; max-width: 980px; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .subtitle {{ color: #555; margin-top: 0; }}
    li {{ margin-bottom: 1rem; }}
    span {{ font-weight: normal; color: #555; }}
    code {{ background: #f1f1f1; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>CineMatch Hybrid Recommender Demo</h1>
  <p class="subtitle">A new user provides a few movie ratings. The system fits a temporary latent user vector and recommends unseen movies.</p>
  <h2>Seed ratings</h2>
  <ul>{seed_items}</ul>
  <h2>Recommendations</h2>
  <ol>{''.join(rec_items)}</ol>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")
