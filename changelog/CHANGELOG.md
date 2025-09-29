# Changelog

All notable changes to the Glovo AI Hybrid Search project will be documented in this file.

## [Unreleased]

### Added - 2025-09-28
- Minimal FastAPI `/chat` endpoint in `src/rag/chat_demo.py` that:
  - Embeds text with JE3 (`src/embeddings/text_je3.py`).
  - Runs pgvector cosine over `glovo_ai.products.text_emb_je3` with `SET LOCAL hnsw.ef_search` via `psycopg2`.
  - Performs Spanish full‑text search (`ts_rank_cd` + `plainto_tsquery`).
  - Fuses results with simple RRF and returns `answer` plus detailed `trace`.
- Updated `requirements.txt` to include `fastapi`, `uvicorn`, `pydantic`.
- Added pytest infrastructure: `pytest` and `openai` added to `requirements.txt` for connectivity tests.
- Tests now auto-load `.env` via `python-dotenv` in `src/tests/test_connectivity.py` and `src/tests/test_openai.py` so credentials are present during pytest runs.
- Ensured `src/rag/config.py` loads `.env` on import to make DB credentials available to the API and utilities.
- Added support for `DATABASE_URL` with `sslmode=require` for Supabase (transaction pooler) connections in `src/rag/config.py` and tests.

### Connection Guidance - 2025-09-28
- Preferred DB config for Supabase:
  - Set `DATABASE_URL` (e.g., `postgresql://user:password@host:port/dbname`), we auto-append `sslmode=require` when missing.
  - Or set discrete envs `user,password,host,port,dbname`; we pass `sslmode=require`.
  - Tests prefer `DATABASE_URL` if present.
- `src/rag/README_rag_mvp.md`: Added minimal MVP task checklist and run steps.
- `src/scripts/simple-supabase-connect.py`: updated to prefer `DATABASE_URL` with `sslmode=require`, printing parsed connection info.
- `src/tests/test_chat_hybrid_logging.py`: pytest that runs multiple queries against `/chat`, saving full trace logs under `log/chat_run_<timestamp>/` (request, response, SQL, candidates, fused list).
- `src/rag/README_rag_mvp.md`: Added Quick Run snippet for `uvicorn` and a detailed explanation of test artifacts:
  - `00_meta.json`
  - `01_request.json`
  - `02_response_status.json`
  - `03_response_json.json`
  - `04_sql.json`
  - `05_candidates_vector_top.json`
  - `06_candidates_lexical_top.json`
  - `07_final_fused.json`
  - Plus guidance on how to interpret these files.

### Fixed - 2025-09-28
- Corrected multi-line SQL assembly in `src/rag/chat_demo.py` lexical query to resolve syntax errors.

### Notes - 2025-09-28
- Tests to validate DB and OpenAI connectivity will run via `pytest`. Consulted FastAPI and pytest docs for best practices; using `fastapi.testclient` for API tests and simple env-driven connectivity checks for external services.

## [0.3.2] - 2025-09-04

### Added
- New evaluation scripts (simple, CLI-first, minimal):
  - `src/tests/search_quality_metrics.py`
    - Metrics: Hybrid Recall@{1,5,10} with structured filters `(country_code, city_code)` using `collection_section` as a relevance proxy; Filter‑Separation score for `collection_section` and `store_name`.
    - CLI: `--limit`, `--out-dir`, `--seed` (seed controls sampling and all randomness).
    - Output: timestamped JSON to `src/tests/metrics/` (user can direct to subfolders via `--out-dir`).
  - `src/tests/relevance_quality_metrics.py`
    - Metrics: 1-NN Accuracy (leave-one-out, cosine), Silhouette Score (cosine), KMeans ARI/NMI (k = unique labels from `collection_section`), Label Consistency@{1,5,10}.
    - CLI: `--limit`, `--out-dir`, `--seed` (reproducible KMeans and neighbor sampling).
    - Output: timestamped JSON to `src/tests/metrics/`.
  - `src/tests/multimodal_alignment_metrics.py`
    - Metrics (cross‑modal): Recall@{1,5,10}, MRR, Positive vs Negative Pair Separation (cosine/euclidean) for aligned text↔image pairs.
    - Columns: prioritizes matched 512D pair `text_emb_clip_multi` ↔ `image_emb_clip`; falls back to `clip_text_emb`/`clip_image_emb` if needed.
    - Safeguards: dimension mismatch detection recorded in output instead of failing.
    - CLI: `--limit`, `--out-dir`, `--seed`.
    - Output: timestamped JSON to `src/tests/metrics/`.

### Changed
- Standardized CLI flags across all evaluation scripts (`--limit`, `--out-dir`, `--seed`) and ensured deterministic behavior when `--seed` is provided.
- Aligned cross‑modal evaluation to use truly matched CLIP spaces (512D) by default: `text_emb_clip_multi` with `image_emb_clip`.

### Deprecated
- Replaced `src/tests/evaluate_supabase_metrics.py` with the three purpose‑built scripts above. That file is no longer used for evaluation.

### Outputs & Reporting
- Results are written as timestamped JSON under `src/tests/metrics/`. Human‑readable summaries can be stored under `src/tests/reports/` (structure mirrors metrics by category).
- Examples now include: `search_quality_metrics_*.json`, `relevance_quality_metrics_*.json`, `multimodal_alignment_metrics_*.json` and corresponding Markdown summaries under `src/tests/reports/`.

### Added
- **New vector embedding models and backfill scripts**
  - Added four new vector columns to `glovo_ai.products` table:
    - `text_emb_e5` (384D) - embeddings from `intfloat/multilingual-e5-small`
    - `text_emb_gte` (768D) - embeddings from `Alibaba-NLP/gte-multilingual-base`
    - `image_emb_clip` (512D) - image embeddings from `openai/clip-vit-base-patch32`
    - `text_emb_clip_multi` (512D) - multilingual text embeddings from `sentence-transformers/clip-ViT-B-32-multilingual-v1`
  - Created embedding modules:
    - `src/embeddings/text_e5_small.py` - E5-small multilingual text embeddings
    - `src/embeddings/text_gte_base.py` - GTE multilingual-base text embeddings
    - `src/embeddings/text_clip_multi.py` - CLIP multilingual text embeddings (512D)
    - `src/embeddings/image_clip_vitb32.py` - CLIP ViT-B/32 image embeddings (512D)
  - Implemented batch backfill scripts with efficient processing:
    - `src/ingest/backfill_text_e5.py` - fills `text_emb_e5` column
    - `src/ingest/backfill_text_gte.py` - fills `text_emb_gte` column
    - `src/ingest/backfill_clip_512.py` - fills both `text_emb_clip_multi` and `image_emb_clip` columns
  - Enhanced `src/ingest/ingest_s3_csv.py`:
    - Added filter to require non-null `product_description` in all records
    - Added optional country code filtering with `--country-codes` parameter (single or multiple codes)
    - Example usage: `--country-codes ES PT IT` or `--country-codes ES`
  - Updated `src/ingest/backfill_text_je3.py` to include `collection_section` in concatenated text
  - All backfill scripts now process only NULL embedding records for efficiency
  - Implemented batch updates using `execute_values` for optimal database performance
  - Added comprehensive column documentation with COMMENT ON COLUMN statements

### Enhanced
- Database schema with ALTER TABLE statements for new vector columns
- Text concatenation format: `product_name · collection_section. product_description`
- Batch processing optimization across all embedding backfill scripts
- NULL-only filtering to prevent reprocessing of existing embeddings

### Technical Improvements
- Cross-modal CLIP embeddings: 512D text and image embeddings in same vector space
- Multilingual support across all new embedding models
- Efficient batch encoding with configurable batch sizes
- Memory-optimized tensor operations with GPU acceleration

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

- **GPU-optimized embedding modules** with CUDA acceleration
  - `src/embeddings/text_je3.py` - Jina Embeddings v3 (1024D multilingual text embeddings)
    - Automatic CUDA/CPU device detection with FP16 optimization
    - PyTorch SDPA attention implementation (avoids flash-attention dependencies)
    - Configurable batch sizes with default batch_size=256 for optimal GPU utilization
    - CUDA memory optimization with expandable segments allocator
    - Model warm-up for kernel stabilization and reduced first-inference latency
    - Intelligent HF_HOME cache directory usage with fallback to `./models_cache`
  - `src/embeddings/clip_v2.py` - Jina CLIP v2 (1024D unified text/image embeddings)
    - Dual-modal support for both text and image embedding generation
    - GPU-optimized with FP16 precision and SDPA attention implementation
    - Separate batch size configurations: 256 for text, 128 for images
    - Comprehensive warm-up strategy with dummy text and image inputs
    - Memory-efficient tensor operations with CPU transfer only at final step
    - Support for PIL Image inputs with automatic preprocessing

- **Embedding backfill scripts** with enhanced performance and monitoring
  - `src/ingest/backfill_text_je3.py` - JE-3 text embedding backfill with tqdm progress bars
  - `src/ingest/backfill_clip_v2.py` - CLIP v2 backfill supporting both text and image modes
    - Dual-mode operation: `--mode text` or `--mode image`
    - Anonymous S3 access for public image datasets
    - Configurable batch sizes optimized for different modalities
    - Real-time progress tracking with tqdm integration
  - In-memory CSV conversion for COPY operations (no temporary files)

- **Google Colab production notebook** (`src/loader/Backfill_JE3_CLIPv2_to_Supabase.ipynb`)
  - Complete GPU-accelerated workflow for embedding generation in Google Colab
  - Tesla T4 GPU optimization with CUDA 12.6 and PyTorch 2.8 compatibility
  - Integrated Google Drive mounting for HF_HOME cache persistence
  - Colab Secrets integration for secure database credential management
  - Step-by-step setup: dependencies, GPU verification, Drive cache, secrets, and execution
  - Production-ready with proper Python package structure and module imports
  - Automated environment configuration for CUDA memory allocation and tokenizer parallelism

## [0.2.0] - 2025-08-24

### Added
- **GPU-accelerated embedding generation** with production-ready CUDA optimization
- **Jina Embeddings v3** and **Jina CLIP v2** integration with intelligent caching
- **Google Colab workflow** for scalable embedding backfill operations
- **HF_HOME cache management** for persistent model storage across sessions

### Enhanced
- Embedding modules with FP16 precision and SDPA attention for Tesla T4 compatibility
- Backfill scripts with configurable batch sizes and real-time progress monitoring
- Memory optimization strategies for large-scale embedding generation

### Technical Improvements
- PyTorch CUDA allocator configuration with expandable segments
- Flash-attention fallback handling for older GPU architectures
- Automated model warm-up for reduced inference latency
- Environment variable standardization for Hugging Face cache directories

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
- **Vector dimensions**: 
  - Text embeddings: JE-3 (`jina-embeddings-v3`, 1024d), E5-small (384d), GTE-base (768d)
  - CLIP embeddings: v2 text/image (`jina-clip-v2`, 1024d each), multilingual text (512d), ViT-B/32 image (512d)
- **Index type**: HNSW with cosine similarity (`vector_cosine_ops`)
- **Index tuning**: m=16, ef_construction=200 (configurable)
- **Search optimization**: Recommended `ef_search=40` for runtime queries
- **Database constraints**: No NOT NULLs, no foreign keys for ingestion speed
- **GPU optimization**: CUDA acceleration with FP16 precision, SDPA attention, expandable memory segments
- **Cache management**: HF_HOME environment variable for persistent model storage
- **Cross-modal search**: 512D CLIP text and image embeddings in unified vector space

### Dependencies
- PostgreSQL with pgvector extension
- Python packages: psycopg2, python-dotenv, pandas, s3fs, sentence-transformers, torch, pillow, tqdm, boto3
- CUDA-compatible GPU (Tesla T4 or equivalent) for optimal performance
- Google Colab for cloud-based GPU acceleration
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
2. ~~Implement embedding generation for `text_emb_je3`, `clip_text_emb`, `clip_image_emb`~~ ✅ **Completed** - GPU-optimized modules with Colab workflow
3. ~~Add additional embedding models (E5, GTE, CLIP multilingual)~~ ✅ **Completed** - Four new embedding models with backfill scripts
4. Develop hybrid search query interface with vector similarity scoring
5. Implement multi-modal search combining text and image embeddings
6. Test end-to-end pipeline with sample dataset and performance benchmarking
7. Deploy search API with FastAPI and integrate with production systems

---

[Unreleased]: https://github.com/username/glovo-ai-hybrid-search/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/username/glovo-ai-hybrid-search/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/username/glovo-ai-hybrid-search/releases/tag/v0.1.0

## [0.3.1] - 2025-09-04

### Added
- Supabase evaluation script `src/tests/evaluate_supabase_metrics.py`:
  - Connects via psycopg2 and loads credentials from `.env` (dotenv).
  - Sampling with `--limit` to evaluate subsets (e.g., 1k/5k/10k).
  - Selectable metrics via `--metrics` (or `all`):
    - Cosine/Euclidean pair stats (text↔image aligned by row)
    - Retrieval: Recall@{1,5,10}, nDCG@{1,5,10}, MRR (single relevant)
    - 1-NN accuracy (leave-one-out, cosine) and Silhouette (cosine)
    - KMeans ARI/NMI (chance-adjusted ARI recommended for interpretation)
    - t-SNE and UMAP 2D projections colored by `collection_section`
  - Outputs thesis-ready JSON with brief interpretations and PNG plots under `src/tests/metrics`.
- Reports for planning and results:
  - `src/tests/reports/Report-Planning.md` – sequenced plan from easy to hard to improve metrics
  - `src/tests/reports/report_20250904.md` – initial results and interpretations (1k sample)

### Changed
- Default metrics output directory set to `src/tests/metrics`.
- Requirements updated to include `scikit-learn`, `matplotlib`, and `umap-learn`.

### Notes
- For cross‑modal retrieval, prefer matched CLIP encoders: `clip_text_emb` ↔ `image_emb_clip`.
- Text preprocessing (normalize, concise product text) planned next per Report-Planning.

## [0.3.0] - 2025-08-31

### Added
- Google Colab notebook flow extended to support full embedding pipeline with ordered execution:
  - Uploader cell accepts: `backfill_text_e5.py`, `backfill_text_gte.py`, `backfill_text_je3.py`, `backfill_clip_512.py` and embedding modules `text_e5_small.py`, `text_gte_base.py`, `text_je3.py`, `text_clip_multi.py`, `image_clip_vitb32.py`.
  - Execution order aligned to demo requirements:
    1) Text→Text embeddings: E5 (384D) → GTE (768D) → JE-3 (1024D)
    2) Cross-modal CLIP: Text→Image (`text_emb_clip_multi`, 512D) then Image→Text (`image_emb_clip`, 512D)
  - Batch sizes tuned for Tesla T4 (16 GB): E5=512, GTE=256, JE-3=128, CLIP text=256, CLIP image=128.

- Data ingestion filters enforced in `src/ingest/ingest_s3_csv.py`:
  - Hardcoded whitelist for Spanish-speaking markets applied by default:
    `['MX','CO','ES','AR','PE','VE','CL','GT','EC','BO','CU','DO','HN','PY','SV','NI','CR','PA','UY','GQ','PR']`
  - Mandatory non-null/non-empty `product_description`.
  - Retains existing constraints: valid non-empty `s3_path` and exclusion of auto stores (`store_name` starting with `AS_`).
  - CLI `--country-codes` still supported; combined with the whitelist (intersection) to avoid out-of-scope markets.

### Changed
- Notebook sanity-import cell updated to reference the new embedding modules and CLIP 512 pipeline; legacy `clip_v2` paths remain tolerated only for back-compat.
- Embedding run cells switched from single JE-3 run to the required ordered sequence (E5 → GTE → JE-3) followed by CLIP text→image, then image→text.

### Verified
- Schema check: `database/migrations/02_tables.sql` already contains the required vector columns and dimensions used by the active backfill scripts:
  - `text_emb_e5 vector(384)`
  - `text_emb_gte vector(768)`
  - `text_emb_je3 vector(1024)`
  - `text_emb_clip_multi vector(512)`
  - `image_emb_clip vector(512)`
  No migration changes needed; columns are actively populated by their respective backfill jobs.

### RAG readiness notes
- Current chunking strategy remains record-level: `product_name · collection_section. product_description` per row; no intra-document chunking or overlap.
- A concrete plan for RAG demo is documented (chunking options, hybrid retrieval with BM25+vectors, optional re-ranking, prompt grounding with citations, guardrails, and evaluation). This does not alter code yet but informs next implementation steps.

### Operational recap
- Migrations: `python database/apply_migrations.py`
- Ingest: `python src/ingest/ingest_s3_csv.py --max-records ... --output-dir data`
- Load: `python src/loader/upload_parquet_to_supabase.py --method auto --data-dir data`
