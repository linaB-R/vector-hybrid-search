-- Create flexible products table for hybrid search (idempotent)
-- No NOT NULLs, no FKs; optimized for ingestion speed

-- Trigger function to auto-update updated_at
CREATE OR REPLACE FUNCTION glovo_ai.tg_set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;

-- Core table
CREATE TABLE IF NOT EXISTS glovo_ai.products (
  id bigserial PRIMARY KEY,
  country_code text,
  city_code text,
  store_name text,
  product_name text,
  collection_section text,
  product_description text,
  s3_path text,
  s3_url text,
  meta jsonb,
  text_emb_je3 vector(1024),
  clip_text_emb vector(1024),
  clip_image_emb vector(1024),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Ensure single trigger exists
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgname = 'products_set_updated_at'
  ) THEN
    CREATE TRIGGER products_set_updated_at
    BEFORE UPDATE ON glovo_ai.products
    FOR EACH ROW
    EXECUTE FUNCTION glovo_ai.tg_set_updated_at();
  END IF;
END$$;




-- New vector columns for additional models (idempotent)
-- Note: Use separate ALTER TABLE statements for compatibility
ALTER TABLE glovo_ai.products
  ADD COLUMN IF NOT EXISTS text_emb_e5 vector(384);

COMMENT ON COLUMN glovo_ai.products.text_emb_e5 IS '384D text embeddings from intfloat/multilingual-e5-small';

ALTER TABLE glovo_ai.products
  ADD COLUMN IF NOT EXISTS text_emb_gte vector(768);

COMMENT ON COLUMN glovo_ai.products.text_emb_gte IS '768D text embeddings from Alibaba-NLP/gte-multilingual-base';

ALTER TABLE glovo_ai.products
  ADD COLUMN IF NOT EXISTS image_emb_clip vector(512);

COMMENT ON COLUMN glovo_ai.products.image_emb_clip IS '512D image embeddings from openai/clip-vit-base-patch32 (CLIP ViT-B/32 image encoder).';

ALTER TABLE glovo_ai.products
  ADD COLUMN IF NOT EXISTS text_emb_clip_multi vector(512);

COMMENT ON COLUMN glovo_ai.products.text_emb_clip_multi IS '512D multilingual text embeddings from sentence-transformers/clip-ViT-B-32-multilingual-v1 (text encoder aligned to CLIP image space). Use cosine similarity against image_emb_clip for cross-modal retrieval.';