# Sample run results

These results were generated with:

```bash
PYTHONPATH=src python -m cinematch.cli all --data data/sample --out outputs_sample --epochs 12 --factors 12 --cold-count 5 --cold-min-ratings 4
```

## Warm-start recommendation

| Model | RMSE | MAE | P@10 | R@10 | NDCG@10 | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Popularity | 1.2698 | 1.0358 | 0.0875 | 0.5625 | 0.2303 | 0.8500 |
| ItemKNN CF | 0.9942 | 0.7657 | 0.1500 | 1.0000 | 0.5247 | 0.9000 |
| Matrix factorization | 1.2707 | 1.0542 | 0.0875 | 0.5625 | 0.2324 | 0.8500 |
| Hybrid MF + ItemKNN | 1.1779 | 0.9689 | 0.1375 | 0.9375 | 0.4141 | 0.9000 |

On the tiny sample data, ItemKNN has the strongest rating prediction and ranking metrics. The hybrid model improves over raw matrix factorization by combining latent factors with neighborhood similarity.

## Cold-start recommendation

| Model | P@10 | R@10 | NDCG@10 | Coverage | Users |
| --- | --- | --- | --- | --- | --- |
| Cold genre profile | 0.2222 | 1.0000 | 0.9110 | 1.0000 | 9 |
| Metadata latent mapper | 0.2222 | 1.0000 | 0.7066 | 1.0000 | 9 |

The cold-start experiment hides selected movies from collaborative training, then ranks those movies using genre profiles and metadata-to-latent predictions.
