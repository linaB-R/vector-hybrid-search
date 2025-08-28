import os
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm

from src.embeddings.text_gte_base import encode_texts


def _db():
    load_dotenv()
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
    )


def _fetch_batch(cur, limit: int) -> List[Tuple[int, str]]:
    cur.execute(
        """
        SELECT id,
               TRIM(BOTH FROM (
                 COALESCE(product_name, '') ||
                 CASE WHEN COALESCE(collection_section,'') <> '' THEN ' · ' || collection_section ELSE '' END ||
                 CASE WHEN COALESCE(product_description,'') <> '' THEN '. ' || product_description ELSE '' END
               )) AS txt
        FROM glovo_ai.products
        WHERE text_emb_gte IS NULL
        ORDER BY id
        LIMIT %s;
        """,
        (limit,),
    )
    return cur.fetchall()


def _vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def main(batch_size: int = 256):
    with _db() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM glovo_ai.products WHERE text_emb_gte IS NULL;
                """
            )
            total = cur.fetchone()[0]
            pbar = tqdm(total=total)
            while True:
                rows = _fetch_batch(cur, batch_size)
                if not rows:
                    break
                ids = [r[0] for r in rows]
                texts = [r[1] for r in rows]
                embs = encode_texts(texts, batch_size=batch_size)
                pairs = [(i, _vec_literal(e)) for i, e in zip(ids, embs)]
                sql = (
                    "UPDATE glovo_ai.products AS p "
                    "SET text_emb_gte = v.emb::vector, updated_at = now() "
                    "FROM (VALUES %s) AS v(id, emb) "
                    "WHERE p.id = v.id;"
                )
                execute_values(cur, sql, pairs, template="(%s,%s)")
                pbar.update(len(rows))
            pbar.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backfill GTE multilingual-base text embeddings")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    main(batch_size=args.batch_size)


