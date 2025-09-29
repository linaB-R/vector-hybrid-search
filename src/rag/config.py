"""
Config & connections for the multimodal RAG MVP.

Environment variables expected (from .env):
  user, password, host, port, dbname, OPENAI_API_KEY

Notes:
- DB connection helper mirrors your psycopg2 sample.
- If you adopt langchain_postgres.PGVector, consider psycopg3 DSN.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
import psycopg2  # You are already using psycopg2 in your sample

# Load .env early so any module importing SETTINGS picks up credentials
load_dotenv()

@dataclass(frozen=True)
class Settings:
    db_user: str = os.getenv("user", "")
    db_password: str = os.getenv("password", "")
    db_host: str = os.getenv("host", "")
    db_port: str = os.getenv("port", "")
    db_name: str = os.getenv("dbname", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Table & columns (adjust to your schema)
    table_fq: str = "glovo_ai.products"
    col_id: str = "id"
    col_text: str = "product_description"
    col_name: str = "product_name"
    col_img_url: str = "s3_url"

    # Vector columns
    col_vec_je3: str = "text_emb_je3"        # 1024-D
    col_vec_clip_text: str = "clip_text_emb" # 1024-D
    col_vec_clip_image: str = "clip_image_emb" # 1024-D

    # Retrieval knobs
    top_k: int = 20
    ef_search_default: int = 40  # runtime HNSW knob
    bm25_limit: int = 60
    vector_limit: int = 60

SETTINGS = Settings()

def get_db_conn():
    """
    Returns a psycopg2 connection using your env variables.
    Intended for:
      - direct SQL (BM25/tsvector, HNSW knobs),
      - text-to-SQL execution (read-only role recommended).
    """
    # Prefer a DATABASE_URL if provided , enforce sslmode=require
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if "sslmode=" not in database_url:
            if "?" in database_url:
                database_url = f"{database_url}&sslmode=require"
            else:
                database_url = f"{database_url}?sslmode=require"
        return psycopg2.connect(database_url)
    # Fallback to discrete fields; add sslmode=require
    return psycopg2.connect(
        user=SETTINGS.db_user,
        password=SETTINGS.db_password,
        host=SETTINGS.db_host,
        port=SETTINGS.db_port,
        dbname=SETTINGS.db_name,
        sslmode="require",
    )

def get_sqlalchemy_psycopg3_url() -> str:
    """
    Optional: If your team adopts langchain_postgres.PGVector (psycopg3),
    return a SQLAlchemy-style DSN (postgresql+psycopg://...).
    """
    # TODO: Build and return DSN string for PGVector via psycopg3
    # Example format (fill with env vars):
    # return f"postgresql+psycopg://{...}:{...}@{...}:{...}/{...}"
    raise NotImplementedError
