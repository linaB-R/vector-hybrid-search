-- Teardown for local/dev re-runs (idempotent best-effort)

-- Drop indexes if exist
DROP INDEX IF EXISTS idx_products_text_emb_hnsw;
DROP INDEX IF EXISTS idx_products_image_emb_hnsw;

-- Drop table
DROP TABLE IF EXISTS glovo_ai.products;

-- Drop trigger function
DROP FUNCTION IF EXISTS glovo_ai.tg_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS glovo_ai;


