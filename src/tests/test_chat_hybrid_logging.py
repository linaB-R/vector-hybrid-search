import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from src.rag.chat_demo import app


load_dotenv()


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_log(run_dir: Path, name: str, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / f"{name}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@pytest.mark.parametrize(
    "query",
    [
        "productos con chocolate sin azúcar",
        "galletas integrales sin azúcar marca x",
        "¿Cuántos productos por colección en Bogotá?",
        "bebidas energéticas sin azúcar",
    ],
)
def test_chat_hybrid_trace_logging(query: str):
    client = TestClient(app)

    ts = _now_ts()
    run_dir = Path("log") / f"chat_run_{ts}"
    meta = {
        "timestamp": ts,
        "env": {
            "DATABASE_URL": bool(os.getenv("DATABASE_URL")),
            "host": os.getenv("host"),
            "dbname": os.getenv("dbname"),
            "OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY")),
        },
        "query": query,
        "params": {"k": 20, "ef_search": 40},
    }
    _write_log(run_dir, "00_meta", meta)

    # Request body
    body = {"text": query, "k": 20, "ef_search": 40}
    _write_log(run_dir, "01_request", body)

    start = time.time()
    resp = client.post("/chat", json=body)
    elapsed_ms = int((time.time() - start) * 1000)

    record = {
        "status_code": resp.status_code,
        "elapsed_ms": elapsed_ms,
    }
    _write_log(run_dir, "02_response_status", record)

    assert resp.status_code == 200, f"/chat failed: {resp.text}"

    data = resp.json()
    _write_log(run_dir, "03_response_json", data)

    # Traceability checks
    assert "answer" in data
    assert "trace" in data
    trace = data["trace"]
    assert "retrieval" in trace
    retr = trace["retrieval"]
    # Save vector and lexical SQL explicitly for audit
    _write_log(run_dir, "04_sql", {
        "vector_sql": retr.get("vector_sql"),
        "lexical_sql": retr.get("lexical_sql"),
        "params": retr.get("params"),
    })

    # Save candidates and final fusion
    _write_log(run_dir, "05_candidates_vector_top", {
        "vector_top": retr.get("vector_top"),
    })
    _write_log(run_dir, "06_candidates_lexical_top", {
        "lexical_top": retr.get("lexical_top"),
    })
    _write_log(run_dir, "07_final_fused", {
        "final": retr.get("final"),
    })

    # Basic content expectations (non-empty most of the time)
    assert isinstance(retr.get("final"), list)


