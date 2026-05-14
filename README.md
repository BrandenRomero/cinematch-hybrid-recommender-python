# CineMatch Hybrid Recommender - Python Final Project

CineMatch is an applied machine learning project that builds a movie recommendation system similar to a lightweight product prototype. It uses MovieLens-style ratings and metadata to produce recommendations, evaluate recommendation quality, and generate a small HTML demo for a new user.

This Python version replaces the earlier JavaScript/Node implementation with a standard Python package under `src/cinematch/`.

## Project goal

The system recommends movies from user ratings while handling two product scenarios:

1. **Warm-start recommendation:** recommend existing movies to users with rating history.
2. **Cold-start item recommendation:** recommend newly added movies using metadata such as genres and title tokens before those movies have collaborative ratings.

The main ML work is biased matrix factorization trained with stochastic gradient descent, plus item-based collaborative filtering and a metadata-to-latent ridge regression model for cold-start movies.

## Repository layout

```text
cinematch-hybrid-recommender-python/
├── data/sample/                   # Tiny MovieLens-style dataset for offline runs
├── docs/                          # Proposal and progress report PDFs
├── scripts/download_movielens100k.py
├── src/cinematch/
│   ├── cli.py                     # Main training/evaluation/demo entry point
│   ├── data.py                    # MovieLens data loader and split handling
│   ├── evaluation.py              # RMSE, MAE, Precision@K, Recall@K, NDCG@K, coverage
│   ├── cold_start.py              # Cold-start split and metadata latent mapper
│   ├── recommender.py             # New-user inference and explanations
│   ├── reporting.py               # Markdown/HTML output
│   └── models/
│       ├── popularity.py          # Smoothed popularity baseline
│       ├── item_knn.py            # Item-based collaborative filtering
│       ├── matrix_factorization.py# SGD matrix factorization
│       └── hybrid.py              # Weighted hybrid re-ranker
└── tests/test_pipeline.py         # Basic reproducibility tests
```

## Required packages

Python 3.10+ is recommended.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The core dependencies are `numpy` and `pandas`. `pytest` is used for the included tests.

## How to run with the included sample data

From the repository root:

```bash
python -m cinematch.cli all --data data/sample --out outputs/sample --epochs 12 --factors 12 --cold-count 5 --cold-min-ratings 4
```

This runs the full prototype:

- loads the sample ratings and movie metadata
- trains popularity, ItemKNN, matrix factorization, and hybrid models
- evaluates warm-start recommendation metrics
- performs a cold-start item experiment
- generates demo recommendation outputs

Generated files appear in `outputs/sample/`:

```text
outputs/sample/
├── dataset_summary.json
├── warm_results.json
├── warm_results.md
├── cold_results.json
├── cold_results.md
├── mf_model.json
├── mf_training_history.csv
├── demo_recommendations.json
└── demo.html
```

Open `outputs/sample/demo.html` in a browser to view the demo.


## Included sample outputs

I included `outputs_sample/` and `docs/RESULTS.md` as a reproducible sample run. The sample run was generated with:

```bash
PYTHONPATH=src python -m cinematch.cli all --data data/sample --out outputs_sample --epochs 12 --factors 12 --cold-count 5 --cold-min-ratings 4
```

These files are not required to rerun the project, but they make it easy to inspect preliminary metrics and the generated HTML demo immediately after cloning.

## Run faster on larger data

The ItemKNN model is useful for comparison but slower on full MovieLens 100K because it computes item-item similarities. To skip it:

```bash
python -m cinematch.cli all --data data/ml-100k --out outputs/ml-100k --fast --epochs 16 --factors 24
```

## Download MovieLens 100K

The repo includes a small sample dataset so it works offline. To use the full MovieLens 100K dataset:

```bash
python scripts/download_movielens100k.py
python -m cinematch.cli all --data data/ml-100k --out outputs/ml-100k --fast --epochs 16 --factors 24
```

## Demo with custom seed ratings

```bash
python -m cinematch.cli demo \
  --data data/sample \
  --out outputs/demo_custom \
  --epochs 12 \
  --factors 12 \
  --seed-ratings "Toy Story (1995):5; Star Wars (1977):5; Fargo (1996):4" \
  --recommendations 5
```

The demo fits a temporary user vector from the seed ratings and ranks unseen movies by predicted rating. Each recommendation includes an explanation based on shared genres and nearest liked seed movie in latent-factor space.

## Run tests

```bash
PYTHONPATH=src pytest -q
```

## Main implementation files

- **Training pipeline:** `src/cinematch/cli.py`
- **Matrix factorization model:** `src/cinematch/models/matrix_factorization.py`
- **ItemKNN collaborative filtering:** `src/cinematch/models/item_knn.py`
- **Cold-start metadata mapper:** `src/cinematch/cold_start.py`
- **New-user demo inference:** `src/cinematch/recommender.py`

## Evaluation metrics

Warm-start models are evaluated using:

- RMSE and MAE for rating prediction
- Precision@K, Recall@K, and NDCG@K for ranking quality
- Coverage to measure how much of the catalog appears in recommendations

Cold-start models are evaluated only on held-out cold items, where their collaborative ratings are removed during training and then used only for evaluation.

## GitHub submission note

After creating a GitHub repository, push this folder and submit a link like:

```text
https://github.com/YOUR_USERNAME/cinematch-hybrid-recommender-python
```
