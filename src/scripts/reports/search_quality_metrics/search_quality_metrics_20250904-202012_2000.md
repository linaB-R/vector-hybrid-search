### What you measured (in plain language)
- **Hybrid Recall@K**: With country and city filters applied, what share of the “right-kind-of items” (same `collection_section`) show up in the top K suggestions for each product. Higher is better.
- **Filter-Separation Score**: Do the embeddings naturally keep similar items together and different items apart?
  - **intra_mean**: average similarity for items in the same group.
  - **inter_mean**: average similarity for items in different groups.
  - **separation**: intra_mean − inter_mean. Bigger gap = cleaner grouping. Measured for `collection_section` and `store_name`.

Notes:
- `queries_evaluated`: how many products had enough comparable items to score.
- `avg_pool_size`: average number of candidates after applying `(country_code, city_code)` filters.

---

### How to read each metric
- **Recall@1**: If you only show 1 suggestion, what fraction of the “right” items you could have shown are actually present? At K=1 this is very strict.
- **Recall@5/10**: If you show 5 or 10 suggestions, how much of the relevant pool do you cover? These reveal how strong your top results are.
- **Separation**:
  - If separation ~0: embeddings don’t clearly distinguish categories or stores.
  - If separation high: embeddings keep categories/stores cleanly apart, which helps filtering, faceting, and avoiding “cross-category” noise.

---

### Your results, explained simply

- Scope: 2,000 products; 468 queries had enough filtered candidates; each query had ~136 candidates on average after `(country, city)` filter.

#### text_emb_e5
- **Hybrid Recall**
  - **Recall@1 = 0.216**: With one suggestion, about 21.6% of the relevant pool is captured.
  - **Recall@5 = 0.464**: With five suggestions, ~46.4% of the relevant pool is covered.
  - **Recall@10 = 0.595**: With ten suggestions, ~59.5% covered.
- **Filter-Separation**
  - `collection_section`: separation = **0.044** (intra 0.860 vs inter 0.816)
  - `store_name`: separation = **0.025** (intra 0.842 vs inter 0.818)
- **Takeaway**: e5 surfaces relevant items early and has the best retrieval coverage among the three. Category/store separation is present but comparatively modest.

#### text_emb_gte
- **Hybrid Recall**
  - **Recall@1 = 0.173**, **Recall@5 = 0.417**, **Recall@10 = 0.542**
- **Filter-Separation**
  - `collection_section`: separation = **0.122** (intra 0.649 vs inter 0.526)
  - `store_name`: separation = **0.081** (intra 0.612 vs inter 0.531)
- **Takeaway**: gte groups items by category/store more cleanly than e5 (bigger separation), but retrieves a bit fewer relevant items in the very top positions.

#### text_emb_je3
- **Hybrid Recall**
  - **Recall@1 = 0.169**, **Recall@5 = 0.382**, **Recall@10 = 0.507**
- **Filter-Separation**
  - `collection_section`: separation = **0.140** (intra 0.639 vs inter 0.499)
  - `store_name`: separation = **0.096** (intra 0.601 vs inter 0.505)
- **Takeaway**: je3 has the strongest category/store separation of all, but the weakest top-K retrieval coverage.

---

### What this means for your use case
- **If you care most about “show me good results fast”** (top-5 or top-10 suggestions): e5 is currently best.
- **If you care most about clean category/store organization and faceted filtering**: je3 (and gte) excel due to higher separation.
- **Balanced strategy**: use e5 for initial retrieval, then leverage store/category filters (or a light re-rank) to benefit from the cleaner grouping that gte/je3 indicate.

---

### Why this beats traditional SQL-only search
- Traditional SQL relies on exact words and simple filters. Your embeddings:
  - Find items that “mean the same thing” even if names/descriptions differ.
  - Respect your operational filters `(country, city)` while still surfacing semantically similar products.
  - Naturally cluster by `collection_section` and `store_name`, helping hybrid search and reducing irrelevant cross-category results.

---

### Practical next steps
- **Pick K for UX**: Your numbers suggest K=5 or K=10 provides strong coverage, especially for e5.
- **Hybrid tuning**: Combine e5 retrieval with strict filters and optional re-ranking by category/store signals.
- **Slice by market**: Check metrics per `country_code` or `store_name` to spot where performance is best/worst.
- **Human spot-checks**: Even without labels, review a handful of queries where e5 vs je3 disagree to refine the blend.

- Strongest retrieval: `text_emb_e5`. Strongest grouping: `text_emb_je3` (then `text_emb_gte`). Use this to choose the right default and filters for your SME-focused workflow.