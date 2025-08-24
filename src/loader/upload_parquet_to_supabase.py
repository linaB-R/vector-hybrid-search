import os
import time
import io
from glob import glob
from typing import List, Optional

import pandas as pd
import psycopg2
from dotenv import load_dotenv


TARGET_TABLE = "glovo_ai.products"
LOAD_COLUMNS = [
    "country_code",
    "city_code",
    "store_name",
    "product_name",
    "collection_section",
    "product_description",
    "s3_path",
    "s3_url",
]


def _get_db_conn():
    load_dotenv()
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
    )


def validate_connectivity() -> None:
    """Quick connectivity and table existence check. Raises on failure."""
    with _get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
            cur.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'glovo_ai' AND table_name = 'products';
                """
            )
            if cur.fetchone() is None:
                raise RuntimeError("Target table glovo_ai.products not found.")

            # Ensure new embedding columns exist
            cur.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = 'glovo_ai'
                  AND table_name = 'products'
                  AND column_name IN ('text_emb_je3','clip_text_emb','clip_image_emb');
                """
            )
            cnt = cur.fetchone()[0]
            if cnt < 3:
                raise RuntimeError("Expected embedding columns text_emb_je3, clip_text_emb, clip_image_emb not found.")


def _discover_parquet_files(data_dir: str = "data", specific_file: Optional[str] = None) -> List[str]:
    if specific_file:
        if os.path.exists(specific_file):
            return [specific_file]
        else:
            raise FileNotFoundError(f"Specified parquet file not found: {specific_file}")
    return sorted(glob(os.path.join(data_dir, "*.parquet")))


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only required columns, add missing as None
    for col in LOAD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[LOAD_COLUMNS]

    # Derive full S3 URL from relative s3_path
    def _make_url(p):
        if isinstance(p, str) and p.strip():
            return f"s3://glovo-products-dataset-d1c9720d/{p}"
        return None
    df["s3_url"] = df["s3_path"].map(_make_url)

    # Trim strings and coalesce empty strings to None
    for col in LOAD_COLUMNS:
        series = df[col]
        df[col] = series.map(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].map(lambda x: None if x == "" else x)
    return df


def _copy_into_postgres(df: pd.DataFrame, batch_size: int = 100000) -> int:
    """COPY via STDIN. Returns inserted row count. Retries simple transient failures."""
    inserted = 0
    attempts = 0
    start = time.time()
    while True:
        try:
            with _get_db_conn() as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    for start_idx in range(0, len(df), batch_size):
                        chunk = df.iloc[start_idx : start_idx + batch_size]
                        buf = io.StringIO()
                        # No header; empty fields become NULL in COPY CSV
                        chunk.to_csv(buf, index=False, header=False)
                        buf.seek(0)
                        cols_sql = ",".join(LOAD_COLUMNS)
                        cur.copy_expert(
                            f"COPY {TARGET_TABLE} ({cols_sql}) FROM STDIN WITH (FORMAT CSV)",
                            buf,
                        )
                        inserted += len(chunk)
            break
        except psycopg2.Error as e:
            attempts += 1
            if attempts >= 3:
                raise
            time.sleep(min(2 ** attempts, 5))
    elapsed = time.time() - start
    print(f"COPY inserted={inserted} rows in {elapsed:.2f}s")
    return inserted


def _supabase_insert(df: pd.DataFrame, batch_size: int = 1000) -> int:
    """Fallback: Supabase PostgREST bulk inserts. Returns inserted row count."""
    from supabase import create_client  # defer import

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables are required for fallback path.")

    client = create_client(url, key)
    inserted = 0
    start = time.time()
    # Use schema-qualified table via client.schema
    table = client.schema("glovo_ai").table("products")
    for start_idx in range(0, len(df), batch_size):
        chunk = df.iloc[start_idx : start_idx + batch_size]
        payload = chunk.where(pd.notnull(chunk), None).to_dict(orient="records")
        attempts = 0
        while True:
            try:
                _ = table.insert(payload).execute()
                inserted += len(payload)
                break
            except Exception:
                attempts += 1
                if attempts >= 3:
                    raise
                time.sleep(min(2 ** attempts, 5))
    elapsed = time.time() - start
    print(f"Supabase inserted={inserted} rows in {elapsed:.2f}s")
    return inserted


def main(method: str = "auto", data_dir: str = "data", copy_batch_size: int = 100000, api_batch_size: int = 1000, parquet_file: Optional[str] = None) -> None:
    files = _discover_parquet_files(data_dir, specific_file=parquet_file)
    if not files:
        print("No parquet files found under data/; nothing to do.")
        return

    validate_connectivity()

    total_rows = 0
    total_inserted = 0
    overall_start = time.time()

    for path in files:
        file_start = time.time()
        df = pd.read_parquet(path)
        df = _normalize(df)
        rows = len(df)
        total_rows += rows
        print(f"Processing {os.path.basename(path)} rows={rows}")

        if method == "copy" or method == "auto":
            try:
                total_inserted += _copy_into_postgres(df, batch_size=copy_batch_size)
                elapsed = time.time() - file_start
                print(f"OK COPY {os.path.basename(path)} in {elapsed:.2f}s")
                continue
            except Exception as e:
                if method == "copy":
                    raise
                print(f"COPY failed, falling back to Supabase insert: {e}")

        total_inserted += _supabase_insert(df, batch_size=api_batch_size)
        elapsed = time.time() - file_start
        print(f"OK Supabase {os.path.basename(path)} in {elapsed:.2f}s")

    overall_elapsed = time.time() - overall_start
    print(
        {
            "rows_processed": total_rows,
            "rows_inserted": total_inserted,
            "seconds": round(overall_elapsed, 2),
            "method": method,
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load local parquet into glovo_ai.products")
    parser.add_argument("--method", choices=["auto", "copy", "supabase"], default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--parquet-file", help="Specific parquet file to load (overrides data-dir discovery)")
    parser.add_argument("--copy-batch-size", type=int, default=100000, help="Rows per COPY chunk")
    parser.add_argument("--api-batch-size", type=int, default=1000, help="Rows per Supabase insert batch")
    args = parser.parse_args()

    main(
        method=args.method,
        data_dir=args.data_dir,
        copy_batch_size=args.copy_batch_size,
        api_batch_size=args.api_batch_size,
        parquet_file=args.parquet_file,
    )

"""
Notes:
- To change batch sizes, pass --copy-batch-size and --api-batch-size.
- To test fallback path, run with --method supabase.
- Staging-table + merge: replace TARGET_TABLE with a temp/staging table, COPY there, then run
  INSERT INTO glovo_ai.products (cols...) SELECT ... FROM staging ON CONFLICT DO NOTHING; when a
  natural dedupe key is defined later.
"""


