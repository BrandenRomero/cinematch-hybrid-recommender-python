from pathlib import Path

from cinematch.data import load_dataset
from cinematch.cold_start import MetadataLatentMapper, create_cold_start_split, evaluate_cold_ranking
from cinematch.evaluation import evaluate_model
from cinematch.models import MatrixFactorizationModel, PopularityModel
from cinematch.recommender import parse_seed_ratings, recommend_for_new_user


def test_load_sample_dataset():
    dataset = load_dataset(Path("data/sample"), seed=7)
    assert dataset.n_users > 0
    assert dataset.n_items > 0
    assert len(dataset.train) > 0
    assert len(dataset.test) > 0


def test_popularity_and_mf_train_on_sample():
    dataset = load_dataset(Path("data/sample"), seed=7)
    popularity = PopularityModel().fit(dataset)
    pop_result = evaluate_model(popularity, dataset, k=5)
    assert pop_result["RMSE"] >= 0
    mf = MatrixFactorizationModel(factors=6, epochs=2, seed=7).fit(dataset, dataset.test)
    mf_result = evaluate_model(mf, dataset, k=5)
    assert mf_result["RMSE"] >= 0
    assert len(mf.history) == 2


def test_demo_recommendations_are_generated():
    dataset = load_dataset(Path("data/sample"), seed=7)
    mf = MatrixFactorizationModel(factors=6, epochs=2, seed=7).fit(dataset, dataset.test)
    seeds = parse_seed_ratings("Toy Story (1995):5; Star Wars (1977):5; Fargo (1996):4", dataset)
    recs = recommend_for_new_user(mf, dataset, seeds, n=3)
    assert len(recs) == 3
    assert all("title" in rec for rec in recs)


def test_metadata_mapper_scores_cold_items():
    dataset = load_dataset(Path("data/sample"), seed=7)
    split = create_cold_start_split(dataset, count=3, min_ratings=4, seed=7)
    warm_mf = MatrixFactorizationModel(factors=6, epochs=2, seed=7).fit(split.warm_dataset, split.warm_dataset.test)
    warm_items = set(split.warm_dataset.train["item"].unique().astype(int))
    mapper = MetadataLatentMapper(vocab_limit=20).fit(split.warm_dataset, warm_mf, warm_items=warm_items)

    result = evaluate_cold_ranking(mapper, dataset, split.cold_items, split.cold_ratings, k=5)

    assert result["users_evaluated"] >= 0
    assert 0.0 <= result["coverage"] <= 1.0
