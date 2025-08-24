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


