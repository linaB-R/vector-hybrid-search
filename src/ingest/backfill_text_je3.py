import os
from typing import List, Tuple

import psycopg2
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.embeddings.text_je3 import encode_texts


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
               COALESCE(product_name, '') || CASE WHEN COALESCE(product_description,'') <> '' THEN '. ' || product_description ELSE '' END AS txt
        FROM glovo_ai.products
        WHERE text_emb_je3 IS NULL
        ORDER BY id
        LIMIT %s;
        """,
        (limit,),
    )
    return cur.fetchall()


def _vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def main(batch_size: int = 128):
    with _db() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM glovo_ai.products
                WHERE text_emb_je3 IS NULL;
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
                embs = encode_texts(texts)
                for _id, emb in zip(ids, embs):
                    cur.execute(
                        "UPDATE glovo_ai.products SET text_emb_je3 = %s::vector, updated_at = now() WHERE id = %s;",
                        (_vec_literal(emb), _id),
                    )
                pbar.update(len(rows))
            pbar.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backfill JE-3 text embeddings")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    main(batch_size=args.batch_size)


