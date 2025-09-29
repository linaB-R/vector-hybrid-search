"""
FastAPI entrypoint exposing POST /chat for minimal text retrieval (SQL full-text + pgvector).

Minimal, direct implementation.
"""

from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel

from .config import SETTINGS, get_db_conn
from ..embeddings.text_je3 import encode_texts

app = FastAPI(title="RAG Chat Demo (Minimal)")


class ChatRequest(BaseModel):
    text: Optional[str] = None
    k: int = SETTINGS.top_k
    ef_search: int = SETTINGS.ef_search_default


class ChatResponse(BaseModel):
    answer: str
    trace: Dict[str, Any]


def _vec_to_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _vector_search(text: str, k: int, ef_search: int) -> Tuple[List[Dict[str, Any]], str]:
    emb = encode_texts([text])[0]
    vec_lit = _vec_to_literal(emb)
    sql = (
        f"SELECT {SETTINGS.col_id} AS id, {SETTINGS.col_name} AS name, "
        f"{SETTINGS.col_text} AS text, {SETTINGS.col_img_url} AS image_url, "
        f"1 - ({SETTINGS.col_vec_je3} <=> %s::vector) AS score "
        f"FROM {SETTINGS.table_fq} "
        f"ORDER BY {SETTINGS.col_vec_je3} <=> %s::vector ASC "
        f"LIMIT %s"
    )
    rows: List[Dict[str, Any]] = []
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))
            cur.execute(sql, (vec_lit, vec_lit, k))
            for r in cur.fetchall():
                rows.append({
                    "id": r[0],
                    "name": r[1],
                    "text": r[2],
                    "image_url": r[3],
                    "score": float(r[4]),
                })
    return rows, sql


def _lexical_search(text: str, limit: int) -> Tuple[List[Dict[str, Any]], str]:
    sql = f"""
        SELECT {SETTINGS.col_id} AS id, {SETTINGS.col_name} AS name,
               {SETTINGS.col_text} AS text, {SETTINGS.col_img_url} AS image_url,
               ts_rank_cd(
                 to_tsvector('spanish', coalesce({SETTINGS.col_name}, '') || ' ' || coalesce({SETTINGS.col_text}, '')),
                 plainto_tsquery('spanish', %s)
               ) AS lex_score
        FROM {SETTINGS.table_fq}
        WHERE to_tsvector('spanish', coalesce({SETTINGS.col_name}, '') || ' ' || coalesce({SETTINGS.col_text}, ''))
              @@ plainto_tsquery('spanish', %s)
        ORDER BY lex_score DESC
        LIMIT %s
    """
    rows: List[Dict[str, Any]] = []
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (text, text, limit))
            for r in cur.fetchall():
                rows.append({
                    "id": r[0],
                    "name": r[1],
                    "text": r[2],
                    "image_url": r[3],
                    "lex_score": float(r[4]),
                })
    return rows, sql


def _rrf_fuse(lex_rows: List[Dict[str, Any]], vec_rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    k1 = 60.0
    rrf: Dict[Any, float] = {}
    by_id: Dict[Any, Dict[str, Any]] = {}
    for rank, r in enumerate(lex_rows, start=1):
        rid = r["id"]
        rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (k1 + rank)
        by_id.setdefault(rid, {}).update(r)
    for rank, r in enumerate(vec_rows, start=1):
        rid = r["id"]
        rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (k1 + rank)
        by_id.setdefault(rid, {}).update(r)
    fused = [dict(by_id[rid], hybrid_score=score) for rid, score in rrf.items()]
    fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return fused[:k]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.text or not req.text.strip():
        return ChatResponse(answer="Debe proporcionar texto de consulta.", trace={"error": "empty_text"})

    vec_rows, vec_sql = _vector_search(req.text.strip(), req.k, req.ef_search)
    lex_rows, lex_sql = _lexical_search(req.text.strip(), SETTINGS.bm25_limit)
    fused = _rrf_fuse(lex_rows, vec_rows, req.k)

    top_names = ", ".join(r["name"] for r in fused[:5]) if fused else "sin resultados"
    answer = f"Resultados top: {top_names}"

    trace = {
        "intent": "hybrid_text",
        "retrieval": {
            "params": {"k": req.k, "ef_search": req.ef_search},
            "vector_sql": vec_sql,
            "lexical_sql": lex_sql,
            "vector_top": vec_rows[:5],
            "lexical_top": lex_rows[:5],
            "final": fused,
        },
    }

    return ChatResponse(answer=answer, trace=trace)

