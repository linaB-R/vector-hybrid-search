# Changelog

All notable changes to the Glovo AI Hybrid Search project will be documented in this file.

## [Unreleased]

### Added
- **Production parquet loader** (`src/loader/upload_parquet_to_supabase.py`)
  - Bulk loads filtered parquet datasets into `glovo_ai.products` table
  - Primary method: PostgreSQL COPY FROM STDIN for maximum performance
  - Fallback method: Supabase Python client bulk inserts in configurable batches
  - Command-line interface with flexible options:
    - `--method`: Choose between "auto" (COPY with fallback), "copy", or "supabase"
    - `--parquet-file`: Load specific parquet file instead of directory discovery
    - `--data-dir`: Specify directory containing parquet files (default: `data/`)
    - `--copy-batch-size`: Rows per COPY chunk (default: 100,000)
    - `--api-batch-size`: Rows per Supabase insert batch (default: 1,000)
  - Data normalization: Trims strings, converts empty strings to NULL
  - Idempotent and restart-safe with structured logging
  - Robust error handling with retry logic for transient failures
  - Connectivity validation before processing
  - Maps parquet columns 1:1 to database schema, leaves embedding fields as NULL
  - Derives and stores full S3 URL in new `s3_url` column for direct downloads
- **Embedding backfill scripts**
  - `src/ingest/backfill_text_je3.py` now shows a tqdm progress bar
  - `src/ingest/backfill_clip_v2.py` now shows tqdm progress bars for text and image modes and reads from public S3 with anonymous access
  - In-memory CSV conversion for COPY operations (no temporary files)

## [0.1.0] - 2025-08-23

### Added
- **Initial project setup** for hybrid search store in Supabase PostgreSQL
- **Database schema design** with flexible `glovo_ai.products` table
  - Supports product metadata from FooDI-ML dataset
  - Includes `jsonb` meta column for extensibility and `s3_url` for direct S3 access
  - Vector columns: `text_emb` (384d) and `image_emb` (512d)
  - Automatic timestamp management with `created_at` and `updated_at`

- **SQL migration system** with idempotent files:
  - `database/migrations/00_enable_extensions.sql` - Enables pgvector extension
  - `database/migrations/01_create_schema.sql` - Creates `glovo_ai` schema
  - `database/migrations/02_tables.sql` - Creates products table with trigger function
  - `database/migrations/03_indexes_hnsw.sql` - HNSW indexes for fast K-NN search
  - `database/migrations/04_security_basics.sql` - Placeholder for future RLS policies
  - `database/migrations/99_teardown.sql` - Cleanup script for development

- **Python migration runner** (`database/apply_migrations.py`)
  - Automated SQL execution in correct order
  - Environment variable support via `.env` file
  - Verification of database setup (extensions, schema, table, indexes)
  - Error handling for empty SQL files and connection issues
  - Progress logging and setup validation

- **Data sampling infrastructure**
  - `data/sample_foodi_dataset.py` - Extracts 50 Spanish-speaking country products
  - `data/sample_foodi_es_50.csv` - Sample dataset with product metadata
  - S3 integration with anonymous access for dataset retrieval; backfill scripts fetch from public S3

### Technical Specifications
- **Vector dimensions**: Text embeddings (`jina-embeddings-v3`, 1024d), CLIP text/image (`jina-clip-v2`, 1024d each)
- **Index type**: HNSW with cosine similarity (`vector_cosine_ops`)
- **Index tuning**: m=16, ef_construction=200 (configurable)
- **Search optimization**: Recommended `ef_search=40` for runtime queries
- **Database constraints**: No NOT NULLs, no foreign keys for ingestion speed

### Dependencies
- PostgreSQL with pgvector extension
- Python packages: psycopg2, python-dotenv, pandas, s3fs
- AWS CLI v2 for S3 access

### Development Environment
- **Project structure** reorganized for clarity:
  - `data/` - Dataset files and data processing scripts
  - `database/` - Database migrations and setup scripts
  - `changelog/` - Project documentation and version history
  - `sandbox/` - Quick experiments and non-critical code
- Virtual environment setup with `venv/`
- Requirements management via `requirements.txt`
- Environment configuration via `.env` file

- **Data ingestion module** (`src/ingest/ingest_s3_csv.py`)
  - Streams large CSV datasets from public S3 bucket using pandas + s3fs
  - Filters out auto-generated stores (store_name starting with `AS_*`)
  - Retains only records with valid image paths (non-empty `s3_path`)
  - Command-line interface with configurable parameters:
    - `--max-records`: Limit number of records to process
    - `--output-dir`: Specify directory for Parquet output (default: `data/`)
    - `--output-file`: Specify complete file path for Parquet output (e.g., `data/custom_filename.parquet`). Overrides `--output-dir` and automatically creates parent directories if needed.
    - Real-time progress bar with tqdm: Shows progress as chunks are processed, displays the target number of records, and updates the count of filtered records after each chunk.
  - Outputs filtered data in Parquet format for efficient storage and loading
  - Uses chunked processing to handle large datasets without memory overflow
  - Proper Python package structure with `__init__.py` files for Docker/API deployment

### Next Steps
1. ~~Load sample data into `glovo_ai.products` table~~ ✅ **Completed** - Production loader implemented
2. Implement embedding generation for `text_emb_je3`, `clip_text_emb`, `clip_image_emb`
3. Develop hybrid search query interface
4. Test end-to-end pipeline with sample dataset

---

[Unreleased]: https://github.com/username/glovo-ai-hybrid-search/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/glovo-ai-hybrid-search/releases/tag/v0.1.0
