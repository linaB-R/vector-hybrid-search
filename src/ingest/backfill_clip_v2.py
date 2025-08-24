import os
from io import BytesIO
from typing import List, Tuple

import boto3
import psycopg2
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
               COALESCE(product_name, '') || CASE WHEN COALESCE(product_description,'') <> '' THEN '. ' || product_description ELSE '' END AS txt
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


def backfill_clip_text(batch_size: int = 128):
    with _db() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
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
                texts = [r[1] for r in rows]
                embs = embed_text(texts)
                for _id, emb in zip(ids, embs):
                    cur.execute(
                        "UPDATE glovo_ai.products SET clip_text_emb = %s::vector, updated_at = now() WHERE id = %s;",
                        (_vec_literal(emb), _id),
                    )
                pbar.update(len(rows))
            pbar.close()


def backfill_clip_image(batch_size: int = 64):
    s3 = _s3()
    with _db() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
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
                embs = embed_images(images)
                for _id, emb in zip(ids, embs):
                    cur.execute(
                        "UPDATE glovo_ai.products SET clip_image_emb = %s::vector, updated_at = now() WHERE id = %s;",
                        (_vec_literal(emb), _id),
                    )
                pbar.update(len(rows))
            pbar.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backfill CLIP-v2 text/image embeddings")
    p.add_argument("--mode", choices=["text", "image"], required=True)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    if args.mode == "text":
        backfill_clip_text(batch_size=args.batch_size)
    else:
        backfill_clip_image(batch_size=args.batch_size)


