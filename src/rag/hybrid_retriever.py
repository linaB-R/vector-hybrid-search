"""
Hybrid retriever: lexical (BM25/tsvector) + vector, then score fusion.

Inputs:
  - query_text: str
Params:
  - bm25_limit (default SETTINGS.bm25_limit)
  - vector_limit (default SETTINGS.vector_limit)
  - fusion: 'rrf' | 'blend' (normalized)  # choose your favorite

Outputs:
  - List[Dict] with unified candidates and a 'hybrid_score'.

Team TODO:
- Create a GIN index on tsvector(product_name, product_description) if not present.
- Implement BM25-like ranking via to_tsvector/to_tsquery (or plainto_tsquery).
- Implement reciprocal rank fusion (RRF) or min-max blend for scores.
"""

from typing import List, Dict, Any
from .config import SETTINGS, get_db_conn
from .vector_store import VectorStoreAdapter

class HybridRetriever:
    def __init__(self, vector_store: VectorStoreAdapter):
        self.vector_store = vector_store

    def retrieve(self, query_text: str, k: int = SETTINGS.top_k,
                 ef_search: int = SETTINGS.ef_search_default,
                 fusion: str = "rrf") -> List[Dict[str, Any]]:
        """
        TODO:
          1) Run lexical search (LIMIT SETTINGS.bm25_limit)
          2) Run vector search (JE3 or CLIP text; LIMIT SETTINGS.vector_limit)
          3) Fuse results (RRF or normalized blend), dedupe by id
          4) Return top-k
        """
        raise NotImplementedError

    # --- Internal helpers ---

    def _lexical_candidates(self, query_text: str, limit: int) -> List[Dict[str, Any]]:
        """
        TODO: Use psycopg2, run a SELECT with to_tsquery/plainto_tsquery,
        produce [{'id':..., 'lex_score': float, 'name':..., 'text':..., 'image_url':...}, ...]
        """
        raise NotImplementedError

    def _fuse(self, lex_rows: List[Dict[str, Any]],
              vec_rows: List[Dict[str, Any]],
              method: str, k: int) -> List[Dict[str, Any]]:
        """TODO: Implement RRF or min-max score blending; return top-k."""
        raise NotImplementedError
