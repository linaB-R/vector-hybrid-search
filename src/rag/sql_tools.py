"""
Text-to-SQL (read-only) tool for the graph.

Design:
- Wrap a safe `run_sql(query: str) -> dict` that the LLM can call.
- Enforce:
    * SELECT-only, schema/table allowlist
    * LIMIT injection (e.g., append LIMIT 200 if absent)
    * statement_timeout (e.g., 8s)
- Return rows + columns + execution metadata for tracing.

Option A: Use psycopg2 directly (simple, matches your sample).
Option B: Use LangChain SQLDatabase & Toolkit for convenience.

Expose:
  - describe_schema() -> str  # curated schema prompt for LLM
  - run_sql(query: str) -> dict
"""

from typing import Dict, Any
from .config import SETTINGS, get_db_conn

class SQLTool:
    def __init__(self):
        # TODO: optionally accept a connection pool / role
        pass

    def describe_schema(self) -> str:
        """
        TODO: Return a concise schema description of allowed tables/columns,
        possibly pre-rendered from your migration files.
        """
        raise NotImplementedError

    def run_sql(self, query: str) -> Dict[str, Any]:
        """
        TODO:
          - Validate query (SELECT-only, allowlist)
          - Open conn; SET LOCAL statement_timeout & row_security
          - Execute; fetch rows up to a max
          - Return {"columns": [...], "rows": [...], "rowcount": n, "elapsed_ms": ...}
        """
        raise NotImplementedError
