## Plan to Improve Embedding Metrics (from easiest to hardest)

1) Align models for cross‑modal retrieval (easy)
- Use matched CLIP encoders for text and image: `clip_text_emb` ↔ `image_emb_clip`.
- **Don't mix** `text_emb_clip_multi` with ViT-B/32 image; they're not jointly trained.
- If needed, re-backfill text with the same CLIP text encoder used for the image model.
- Rerun metrics with `--metrics retrieval,cosine,euclidean` and compare vs baseline.
- Expected: larger cosine gap (pos>neg), higher R@1/R@10 and MRR.

2) Per‑section diagnostics (easy)
- Compute Recall@K by `collection_section` to identify strong/weak categories.
- Action: extend script later to group retrieval by label; prioritize weakest sections.
- Find which categories are already separable vs problematic.

3) Increase sample size and top‑K (easy)
- Run `--limit 5000` (or 10000) and add R@50 for stability and ranking curve.
- Balance samples per `collection_section` to stabilize estimates.

4) Text preprocessing (easy→medium)
- Build text fed to CLIP as: `product_name. product_description.`
- **Text quality improvements:**
  - Lowercase, strip punctuation, remove boilerplate/store names
  - Avoid store/boilerplate tokens, keep user-visible product semantics
  - Normalize language while preserving product semantics
- **Prompting trick for CLIP:** prefix with "a product photo of " for CLIP text encoder
- Re‑embed text where needed; store in a new column to avoid overwriting.

5) Model sweep for text‑only structure (medium)
- For label structure metrics (1‑NN, silhouette, ARI/NMI): compare `text_emb_gte`, `text_emb_e5`, `text_emb_je3`.
- Use CLIP↔CLIP only for cross‑modal retrieval.
- Report which best separates `collection_section` after preprocessing.
- **Note:** Negative silhouette and low 1‑NN mean `collection_section` doesn't align well with embedding geometry. Consider coarser or cleaned labels.

6) Dimensionality reduction for clustering stability (medium)
- Apply PCA/SVD to 64–256 dims before KMeans; then compute ARI/NMI.
- **Rationale:** KMeans is more stable in lower dims; improves stability and interpretability.
- Use ARI as the primary chance-adjusted score.
- Don't compare silhouette across different spaces.

7) Two‑stage retrieval (medium→hard)
- Stage 1: CLIP retrieval (cosine) to top‑50/100 candidates.
- Stage 2: re‑rank with a stronger cross‑modal model (e.g., BLIP ITM / SigLIP score).
- Evaluate uplift on R@1/MRR.

8) Index/search tuning (hard, production‑oriented)
- In pgvector, raise `ef_search` at query time; verify recall/latency trade‑off.
- Ensure `vector_cosine_ops` and normalized vectors are consistent end‑to‑end.
- Keep `vector_cosine_ops` consistent with cosine normalization.

9) Data balancing and deduping (hard)
- Balance per‑label sampling for evaluation; filter duplicates/near‑duplicates.
- Duplicate/near-duplicate filtering can reduce noise and improve pair separation.
- Re‑measure to ensure robustness.

**Expected Outcomes:**
- Increase cosine gap (pos>neg)
- Push R@1/R@10 and MRR up
- Improve ARI on reduced-dim features
- Better alignment between `collection_section` labels and embedding geometry

Milestones
- M1: CLIP text↔image alignment + diagnostics (today)
- M2: Text preprocessing + rerun metrics (next)
- M3: Model sweep + PCA KMeans (this week)
- M4: Two‑stage re‑ranker (next week)


