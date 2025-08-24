-- HNSW indexes for fast K-NN on text and image embeddings (idempotent)
-- Trade-offs: HNSW offers higher recall-speed than IVFFlat but uses more memory and
-- has slower build times. Tune m (graph degree) and ef_construction to balance quality/speed.
-- Recommended session for queries: SET maintenance_work_mem TO '1GB';
-- Recommended runtime search tweak: SET LOCAL hnsw.ef_search = 40; -- increase for better recall

-- Jina Embeddings v3 text (cosine)
CREATE INDEX IF NOT EXISTS idx_products_text_je3_hnsw
ON glovo_ai.products
USING hnsw (text_emb_je3 vector_cosine_ops)
WITH (
  m = 16,
  ef_construction = 200
);

-- Jina CLIP v2 text (cosine)
CREATE INDEX IF NOT EXISTS idx_products_clip_text_hnsw
ON glovo_ai.products
USING hnsw (clip_text_emb vector_cosine_ops)
WITH (
  m = 16,
  ef_construction = 200
);

-- Jina CLIP v2 image (cosine)
CREATE INDEX IF NOT EXISTS idx_products_clip_image_hnsw
ON glovo_ai.products
USING hnsw (clip_image_emb vector_cosine_ops)
WITH (
  m = 16,
  ef_construction = 200
);


