### How to read your results (relevance_quality_metrics_20250904-234321_2000.json)

Across 2,000 samples using `collection_section` as the proxy label:

- text_emb_e5
  - 1-NN accuracy: 0.181
  - Silhouette (cosine): -0.1006
  - ARI: 0.0562
  - NMI: 0.8876
  - Label consistency: @1=0.181, @5=0.1243, @10=0.1066

- text_emb_gte
  - 1-NN accuracy: 0.151
  - Silhouette (cosine): -0.1285
  - ARI: 0.0570
  - NMI: 0.8866
  - Label consistency: @1=0.151, @5=0.1099, @10=0.0912

- text_emb_je3
  - 1-NN accuracy: 0.1575
  - Silhouette (cosine): -0.1413
  - ARI: 0.0426
  - NMI: 0.8835
  - Label consistency: @1=0.1575, @5=0.1004, @10=0.0864

What each metric means, in plain terms:
- 1-NN accuracy
  - What it is: For each product, check if its single closest neighbor (by embedding) has the same `collection_section`.
  - Your numbers: 15–18%. That means only about 1 in 6 nearest neighbors is from the same category. This is modest local grouping.
- Silhouette (cosine)
  - What it is: Measures how tightly items cluster by `collection_section` and how far they are from other categories. Range: -1 to 1. Higher is better; negative means overlaps.
  - Your numbers: -0.10 to -0.14. Clusters overlap; categories are not well separated in the embedding space.
- ARI (Adjusted Rand Index)
  - What it is: Agreement between KMeans clusters and `collection_section` (corrected for chance). Range: -1 to 1. Higher is better.
  - Your numbers: ~0.04–0.06. Weak match between learned clusters and categories.
- NMI (Normalized Mutual Information)
  - What it is: Shared information between clusters and categories. Range: 0 to 1. Higher is better.
  - Your numbers: ~0.88. High NMI can co-exist with low ARI when clusters capture some broad category structure (information overlap) but boundaries don’t align cleanly (low agreement on exact assignments).
- Label Consistency @ K
  - What it is: For each product, the fraction of the top-K nearest items that share its `collection_section`.
  - Your numbers: At K=1 it equals 1-NN; by K=10 consistency drops to ~9–11%. The top neighbor occasionally matches; as you look deeper, mixtures increase.

What to take away:
- e5 has the strongest local retrieval signal among the three (best 1-NN and label consistency), but all three show overlapping categories (negative silhouette) and weak one-to-one alignment with `collection_section` (low ARI).
- The high NMI suggests the embeddings still carry category information, but categories aren’t cleanly separated; items straddle boundaries or categories are internally diverse.

Actionable suggestions:
- Use embeddings primarily for semantic similarity, then reinforce with filters (country/city/store) and lightweight re-ranking to boost category coherence.
- If `collection_section` is noisy or broad, consider using finer or cleaner proxy labels (or combine text fields) for evaluation.
- For better cluster separation, try L2-normalized vectors (already applied here), try a different text column (titles vs descriptions), or consider domain-adapted fine-tuning if possible.
- Pair with your Hybrid metrics (you ran those too): those showed stronger practical utility in filtered search scenarios, which is where SMEs benefit most.