"""
Vector store adapters for three retrieval spaces against the same table:
  - text_je3        -> text_emb_je3
  - clip_text       -> clip_text_emb
  - clip_image      -> clip_image_emb

Two operation modes:
  (A) Direct SQL with pgvector operators via psycopg2
  (B) LangChain PGVector (langchain_postgres) using psycopg3 DSN

Team TODO:
- Choose A or B (or support both behind the same interface).
- Plug in your query-time embedding functions from src/embeddings/*.
- Normalize scores as needed (1 - cosine_distance).
"""

from typing import List, Dict, Any, Optional, Tuple
from .config import SETTINGS, get_db_conn, get_sqlalchemy_psycopg3_url

class VectorStoreAdapter:
    """
    Common interface so the graph can call:
      .search_by_text(query_text: str, k: int, ef_search: Optional[int]) -> List[Dict]
      .search_by_image(image_bytes: bytes, k: int, ef_search: Optional[int]) -> List[Dict]
    Each result Dict should include at least:
      {"id": ..., "score": float, "name": ..., "text": ..., "image_url": ...}
    """

    # --- Public API (called by the graph) ---

    def search_text_je3(self, query_text: str, k: Optional[int] = None,
                        ef_search: Optional[int] = None) -> List[Dict[str, Any]]:
        """TODO: embed text with JE3 -> query SETTINGS.col_vec_je3 -> return top K rows."""
        raise NotImplementedError

    def search_clip_text(self, query_text: str, k: Optional[int] = None,
                         ef_search: Optional[int] = None) -> List[Dict[str, Any]]:
        """TODO: embed text with CLIP text -> query SETTINGS.col_vec_clip_text."""
        raise NotImplementedError

    def search_clip_image(self, image_bytes: bytes, k: Optional[int] = None,
                          ef_search: Optional[int] = None) -> List[Dict[str, Any]]:
        """TODO: embed image with CLIP image -> query SETTINGS.col_vec_clip_image."""
        raise NotImplementedError

    # --- Internal helpers your team can implement ---

    def _embed_text_je3(self, text: str) -> List[float]:
        """TODO: call src/embeddings/text_je3.py to get 1024-D vector."""
        raise NotImplementedError

    def _embed_text_clip(self, text: str) -> List[float]:
        """TODO: call src/embeddings/text_clip_multi.py or CLIP v2 text."""
        raise NotImplementedError

    def _embed_image_clip(self, image_bytes: bytes) -> List[float]:
        """TODO: call src/embeddings/image_clip_vitb32.py or CLIP v2 image."""
        raise NotImplementedError

    def _pgvector_sql_search(self, query_vec: List[float], vec_col: str,
                             k: int, ef_search: Optional[int]) -> List[Dict[str, Any]]:
        """
        TODO:
          - open psycopg2 conn
          - SET LOCAL hnsw.ef_search = {ef_search or SETTINGS.ef_search_default}
          - SELECT id, product_name, product_description, image_url, (1 - (vec_col <=> query_vec)) as score
          - ORDER BY (vec_col <=> query_vec) ASC LIMIT k
        """
        raise NotImplementedError

    def _pgvector_langchain_search(self, query_vec: List[float], vec_col: str,
                                   k: int) -> List[Dict[str, Any]]:
        """
        TODO: Alternative path if using langchain_postgres.PGVector (psycopg3).
        - Create PGVector store bound to SETTINGS.table_fq and vec_col
        - Similarity search to top-k
        """
        raise NotImplementedError
