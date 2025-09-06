import os
from io import BytesIO
from typing import List, Tuple
import re
import unicodedata

import boto3
import psycopg2
from psycopg2.extras import execute_values
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

from src.embeddings.clip_v2 import embed_images, embed_text


def _db():
    load_dotenv()
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
    )


def _s3():
    # Anonymous access to public Glovo FooDI-ML dataset bucket
    from botocore import UNSIGNED
    from botocore.config import Config
    
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return s3


def _parse_s3_url(s3_url: str) -> Tuple[str, str]:
    # Expect s3://bucket/key
    if not s3_url.startswith("s3://"):
        # fallback: treat as relative key
        return "glovo-products-dataset-d1c9720d", s3_url
    parts = s3_url.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


def _fetch_text_batch(cur, limit: int):
    cur.execute(
        """
        SELECT id,
               TRIM(BOTH FROM (
                 COALESCE(product_name, '') ||
                 CASE WHEN COALESCE(store_name,'') <> '' THEN ' · ' || store_name ELSE '' END ||
                 CASE WHEN COALESCE(collection_section,'') <> '' THEN ' · ' || collection_section ELSE '' END ||
                 CASE WHEN COALESCE(product_description,'') <> '' THEN '. ' || product_description ELSE '' END
               )) AS txt
        FROM glovo_ai.products
        WHERE clip_text_emb IS NULL
        ORDER BY id
        LIMIT %s;
        """,
        (limit,),
    )
    return cur.fetchall()


def _fetch_image_batch(cur, limit: int):
    cur.execute(
        """
        SELECT id, COALESCE(s3_url, CONCAT('s3://glovo-products-dataset-d1c9720d/', s3_path)) AS s3_url
        FROM glovo_ai.products
        WHERE clip_image_emb IS NULL AND (s3_path IS NOT NULL AND s3_path <> '')
        ORDER BY id
        LIMIT %s;
        """,
        (limit,),
    )
    return cur.fetchall()


def _vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = s.replace("_", " ").replace("/", " ").replace("-", " ")
    kept = []
    for ch in s:
        if ch.isalpha() or ch.isspace():
            kept.append(ch)
    s = "".join(kept)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def backfill_clip_text(batch_size: int = 128, reset: bool = False):
    with _db() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            if reset:
                cur.execute(
                    """
                    UPDATE glovo_ai.products
                    SET clip_text_emb = NULL, updated_at = now()
                    WHERE clip_text_emb IS NOT NULL;
                    """
                )
            cur.execute(
                """
                SELECT COUNT(*)
                FROM glovo_ai.products
                WHERE clip_text_emb IS NULL;
                """
            )
            total = cur.fetchone()[0]
            pbar = tqdm(total=total)
            while True:
                rows = _fetch_text_batch(cur, batch_size)
                if not rows:
                    break
                ids = [r[0] for r in rows]
                texts = [_normalize_text(r[1]) for r in rows]
                embs = embed_text(texts, batch_size=batch_size)
                pairs = [(i, _vec_literal(e)) for i, e in zip(ids, embs)]
                sql = (
                    "UPDATE glovo_ai.products AS p "
                    "SET clip_text_emb = v.emb::vector, updated_at = now() "
                    "FROM (VALUES %s) AS v(id, emb) "
                    "WHERE p.id = v.id;"
                )
                execute_values(cur, sql, pairs, template="(%s,%s)")
                pbar.update(len(rows))
            pbar.close()


def backfill_clip_image(batch_size: int = 64, reset: bool = False):
    s3 = _s3()
    with _db() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            if reset:
                cur.execute(
                    """
                    UPDATE glovo_ai.products
                    SET clip_image_emb = NULL, updated_at = now()
                    WHERE clip_image_emb IS NOT NULL;
                    """
                )
            cur.execute(
                """
                SELECT COUNT(*)
                FROM glovo_ai.products
                WHERE clip_image_emb IS NULL AND (s3_path IS NOT NULL AND s3_path <> '');
                """
            )
            total = cur.fetchone()[0]
            pbar = tqdm(total=total)
            while True:
                rows = _fetch_image_batch(cur, batch_size)
                if not rows:
                    break
                ids = [r[0] for r in rows]
                urls = [r[1] for r in rows]
                images: List[Image.Image] = []
                for s3_url in urls:
                    bkt, key = _parse_s3_url(s3_url)
                    obj = s3.get_object(Bucket=bkt, Key=key)
                    body = obj["Body"].read()
                    img = Image.open(BytesIO(body)).convert("RGB")
                    images.append(img)
                embs = embed_images(images, batch_size=batch_size)
                pairs = [(i, _vec_literal(e)) for i, e in zip(ids, embs)]
                sql = (
                    "UPDATE glovo_ai.products AS p "
                    "SET clip_image_emb = v.emb::vector, updated_at = now() "
                    "FROM (VALUES %s) AS v(id, emb) "
                    "WHERE p.id = v.id;"
                )
                execute_values(cur, sql, pairs, template="(%s,%s)")
                pbar.update(len(rows))
            pbar.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backfill CLIP-v2 text/image embeddings")
    p.add_argument("--mode", choices=["text", "image"], required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--reset", action="store_true", help="Nullify target column before backfill")
    args = p.parse_args()
    if args.mode == "text":
        backfill_clip_text(batch_size=args.batch_size, reset=args.reset)
    else:
        backfill_clip_image(batch_size=args.batch_size, reset=args.reset)


