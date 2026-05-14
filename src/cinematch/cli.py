from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from cinematch.cold_start import (
    GenreProfileColdStart,
    MetadataLatentMapper,
    create_cold_start_split,
    evaluate_cold_ranking,
)
from cinematch.data import load_dataset, summarize_dataset
from cinematch.evaluation import evaluate_model, format_metric_rows
from cinematch.models import HybridRecommender, ItemKNNModel, MatrixFactorizationModel, PopularityModel
from cinematch.recommender import fallback_seed_ratings, parse_seed_ratings, recommend_for_new_user
from cinematch.reporting import render_demo_html, results_markdown
from cinematch.utils import ensure_dir, format_number, markdown_table, write_json

DEFAULT_SEEDS = "Toy Story (1995):5; Twelve Monkeys (1995):5; Star Wars (1977):5; Fargo (1996):5; Contact (1997):4"


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["epoch", "train_rmse", "validation_rmse"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def train_warm_models(dataset, args):
    results = []
    fitted_models = []
    popularity = PopularityModel(m=args.pop_m).fit(dataset)
    fitted_models.append(popularity)
    results.append(evaluate_model(popularity, dataset, k=args.k))

    item_knn = None
    if not args.fast:
        item_knn = ItemKNNModel(neighbors=args.neighbors, min_common=2, shrinkage=10.0).fit(dataset)
        fitted_models.append(item_knn)
        results.append(evaluate_model(item_knn, dataset, k=args.k))

    mf = MatrixFactorizationModel(
        factors=args.factors,
        epochs=args.epochs,
        lr=args.lr,
        reg=args.reg,
        seed=args.seed,
    ).fit(dataset, validation_ratings=dataset.test)
    fitted_models.append(mf)
    results.append(evaluate_model(mf, dataset, k=args.k))

    if item_knn is None:
        hybrid = HybridRecommender([mf, popularity], [0.85, 0.15], name="Hybrid MF + popularity")
    else:
        hybrid = HybridRecommender([mf, item_knn, popularity], [0.70, 0.20, 0.10], name="Hybrid MF + ItemKNN")
    fitted_models.append(hybrid)
    results.append(evaluate_model(hybrid, dataset, k=args.k))
    return {"models": fitted_models, "results": results, "mf": mf}


def run_cold_start(dataset, args):
    split = create_cold_start_split(
        dataset,
        count=args.cold_count,
        min_ratings=args.cold_min_ratings,
        seed=args.seed + 100,
    )
    if not split.cold_items:
        return {"results": [], "message": "No cold-start items met the minimum-rating threshold."}
    warm_mf = MatrixFactorizationModel(
        name="Warm MF for cold-start mapper",
        factors=args.factors,
        epochs=max(6, int(args.epochs * 0.75)),
        lr=args.lr,
        reg=args.reg,
        seed=args.seed + 200,
    ).fit(split.warm_dataset, validation_ratings=split.warm_dataset.test)
    genre_model = GenreProfileColdStart().fit(split.warm_dataset)
    warm_items = set(split.warm_dataset.train["item"].unique().astype(int))
    mapper = MetadataLatentMapper(ridge_lambda=args.mapper_lambda, vocab_limit=args.vocab_limit).fit(
        split.warm_dataset,
        warm_mf,
        warm_items=warm_items,
    )
    results = [
        evaluate_cold_ranking(genre_model, dataset, split.cold_items, split.cold_ratings, k=args.k),
        evaluate_cold_ranking(mapper, dataset, split.cold_items, split.cold_ratings, k=args.k),
    ]
    return {
        "results": results,
        "cold_items": sorted(split.cold_items),
        "cold_ratings": int(len(split.cold_ratings)),
        "warm_mf_history": warm_mf.history,
    }


def save_warm_outputs(out_dir: Path, dataset_summary: dict[str, Any], warm: dict[str, Any], k: int) -> None:
    ensure_dir(out_dir)
    write_json(out_dir / "dataset_summary.json", dataset_summary)
    write_json(out_dir / "warm_results.json", warm["results"])
    write_json(out_dir / "mf_model.json", warm["mf"].to_dict())
    write_history_csv(out_dir / "mf_training_history.csv", warm["mf"].history)
    rows = format_metric_rows(warm["results"], k)
    cols = ["Model", "RMSE", "MAE", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage"]
    (out_dir / "warm_results.md").write_text(results_markdown(rows, cols, "Warm-start results"), encoding="utf-8")


def save_cold_outputs(out_dir: Path, cold: dict[str, Any], k: int) -> None:
    if not cold.get("results"):
        return
    write_json(out_dir / "cold_results.json", cold["results"])
    write_json(out_dir / "cold_split.json", {"cold_items": cold.get("cold_items", []), "cold_ratings": cold.get("cold_ratings", 0)})
    rows = []
    for result in cold["results"]:
        rows.append({
            "Model": result.get("model", ""),
            f"P@{k}": format_number(result.get(f"P@{k}")),
            f"R@{k}": format_number(result.get(f"R@{k}")),
            f"NDCG@{k}": format_number(result.get(f"NDCG@{k}")),
            "Coverage": format_number(result.get("coverage")),
            "Users": result.get("users_evaluated", 0),
        })
    cols = ["Model", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage", "Users"]
    (out_dir / "cold_results.md").write_text(results_markdown(rows, cols, "Cold-start results"), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    dataset = load_dataset(args.data, split=args.split, seed=args.seed)
    summary = summarize_dataset(dataset)
    print(f"Loaded dataset: {summary}")

    warm = train_warm_models(dataset, args)
    save_warm_outputs(out_dir, summary, warm, args.k)
    warm_rows = format_metric_rows(warm["results"], args.k)
    print(markdown_table(warm_rows, ["Model", "RMSE", "MAE", f"P@{args.k}", f"R@{args.k}", f"NDCG@{args.k}", "Coverage"]))

    if args.command in {"all", "cold-start"}:
        cold = run_cold_start(dataset, args)
        save_cold_outputs(out_dir, cold, args.k)
        if cold.get("results"):
            print((out_dir / "cold_results.md").read_text(encoding="utf-8"))
        else:
            print(cold.get("message", "No cold-start results."))

    if args.command in {"all", "demo"}:
        seed_ratings = parse_seed_ratings(args.seed_ratings, dataset)
        if len(seed_ratings) < 2:
            seed_ratings = fallback_seed_ratings(dataset)
        recommendations = recommend_for_new_user(warm["mf"], dataset, seed_ratings, n=args.recommendations)
        demo_dir = Path(args.demo_out) if args.demo_out else out_dir
        ensure_dir(demo_dir)
        write_json(demo_dir / "demo_recommendations.json", {"seed_ratings": seed_ratings, "recommendations": recommendations})
        render_demo_html(seed_ratings, recommendations, demo_dir / "demo.html")
        print(f"Demo files written to {demo_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CineMatch hybrid movie recommender")
    parser.add_argument("command", nargs="?", default="all", choices=["all", "train-warm", "cold-start", "demo"], help="Pipeline stage to run")
    parser.add_argument("--data", default="data/sample", help="MovieLens-style data directory")
    parser.add_argument("--out", default="outputs/sample", help="Output directory")
    parser.add_argument("--demo-out", default=None, help="Optional separate demo output directory")
    parser.add_argument("--split", default="u1", help="MovieLens predefined split prefix, if files exist")
    parser.add_argument("--k", type=int, default=10, help="Ranking cutoff")
    parser.add_argument("--factors", type=int, default=24, help="Matrix factorization latent dimensions")
    parser.add_argument("--epochs", type=int, default=16, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.015, help="SGD learning rate")
    parser.add_argument("--reg", type=float, default=0.045, help="L2 regularization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pop-m", type=float, default=25.0, help="Popularity smoothing strength")
    parser.add_argument("--neighbors", type=int, default=30, help="ItemKNN neighbor count")
    parser.add_argument("--fast", action="store_true", help="Skip ItemKNN for faster full-data runs")
    parser.add_argument("--cold-count", type=int, default=160, help="Number of cold items to hold out")
    parser.add_argument("--cold-min-ratings", type=int, default=15, help="Minimum ratings required for a cold item")
    parser.add_argument("--mapper-lambda", type=float, default=2.0, help="Ridge penalty for metadata-to-latent mapper")
    parser.add_argument("--vocab-limit", type=int, default=120, help="Maximum title-token features")
    parser.add_argument("--seed-ratings", default=DEFAULT_SEEDS, help="Semicolon-separated demo seed ratings")
    parser.add_argument("--recommendations", type=int, default=10, help="Number of demo recommendations")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
