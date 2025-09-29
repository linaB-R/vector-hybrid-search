### What this file measures (in simple terms)
- **Cross‑modal retrieval**: Given a product’s text, find its image (and vice‑versa).
- You have 2,000 text↔image pairs evaluated, using `text_emb_clip_multi` with `image_emb_clip`.

### Your numbers, explained with quick examples

- **Recall@K (text→image)**
  - **Recall@1 = 0.055**: If you make 100 searches (text to find its exact image), about 5–6 times the correct image is ranked #1.
  - **Recall@5 = 0.134**: In the top 5 results, you find the correct image about 13–14 times out of 100.
  - **Recall@10 = 0.199**: In the top 10 results, you find it about 20 times out of 100.
  - Example: If a user searches “red cola can 330ml” (text), the exact product image shows up in the top‑10 about 1 in 5 searches.

- **Recall@K (image→text)**
  - **Recall@1 = 0.064**, **@5 = 0.1525**, **@10 = 0.2065**: Very similar story in the reverse direction (image to find its exact text).

- **MRR (how early the correct match appears)**
  - **Text→Image MRR = 0.106**, **Image→Text MRR = 0.116**
  - Rule of thumb: MRR ~0.10 means the correct match typically appears around rank ~9–10 on average.
  - Example: If you run many searches, expect the right item often to land around position 9 or 10.

- **Positive vs Negative Pair Separation (how well the model distinguishes true pairs from mismatches)**
  - Cosine similarity (higher is better): **positives 0.275 vs negatives 0.223 → gap 0.051**
  - Euclidean distance (lower is better): **positives 1.204 vs negatives 1.246 → gap 0.042**
  - Interpretation: True text↔image pairs are, on average, a bit closer than random mismatches. The gaps are real but modest, so the model separates positives from negatives, though not strongly.
  - Example: Think of scoring pairs: a correct “pizza photo ↔ pizza description” gets ~0.275 similarity; a wrong pair like “pizza photo ↔ shampoo description” gets ~0.223. The difference helps ranking but isn’t huge.

### What this means for you
- You’re doing something normal SQL can’t: matching across modalities (text↔image). The system does retrieve the right pair, but it often appears deeper in the list (top‑10 ~20%).
- The separation scores show the embeddings capture cross‑modal alignment, but the signal is moderate. That’s expected because `text_emb_clip_multi` isn’t jointly trained with your `image_emb_clip` model the way classic CLIP pairs are.

### Quick, practical ways to improve
- Use a truly paired CLIP text model with your `image_emb_clip` (same family) to boost alignment.
- Re‑rank the top‑50 by a stronger cross‑modal scorer (even a simple weighted blend) to lift Recall@5/10.
- Keep structured filters (country/city/store) active: they reduce noise and improve practical relevance.