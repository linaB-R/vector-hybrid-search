import os
import argparse
import pandas as pd
from tqdm import tqdm

# Public S3 CSV source (anonymous access)
S3_CSV_URL = "s3://glovo-products-dataset-d1c9720d/glovo-foodi-ml-dataset.csv"

# Default output directory and filename
DEFAULT_OUTPUT_DIR = "data"
OUTPUT_FILENAME = "filtered_dataset.parquet"

# Rows per chunk when streaming from S3
CHUNK_SIZE = 200_000


def ingest_s3_csv_to_parquet(max_records=None, output_dir=DEFAULT_OUTPUT_DIR, output_file=None, country_codes=None):
    """
    Stream a CSV from public S3 in chunks, filter out rows where `store_name` starts
    with 'AS_' and keep only rows with a non-empty `s3_path`, then write the result
    to a local Parquet file.

    Uses pandas + s3fs for anonymous S3 access and DataFrame.to_parquet for output.
    
    Args:
        max_records: Maximum number of filtered records to keep. If None, keeps all.
        output_dir: Directory where to save the Parquet file. Defaults to 'data'.
        output_file: Complete file path (e.g., 'data/custom.parquet'). Overrides output_dir.
    """
    if output_file:
        output_path = output_file
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    else:
        output_path = os.path.join(output_dir, OUTPUT_FILENAME)
        os.makedirs(output_dir, exist_ok=True)

    filtered_chunks = []
    seen_columns = None
    total_filtered = 0

    # Create CSV reader iterator
    csv_reader = pd.read_csv(
        S3_CSV_URL,
        storage_options={"anon": True},
        chunksize=CHUNK_SIZE,
    )

    # Set up progress bar description
    progress_desc = f"Processing chunks (target: {max_records or 'all'} records)"
    
    with tqdm(desc=progress_desc, unit="chunks") as pbar:
        for chunk in csv_reader:
            if seen_columns is None:
                seen_columns = list(chunk.columns)

            mask_valid_image = chunk["s3_path"].notna() & (
                chunk["s3_path"].astype(str).str.strip() != ""
            )
            # require non-null, non-empty product_description
            mask_has_description = chunk["product_description"].notna() & (
                chunk["product_description"].astype(str).str.strip() != ""
            )
            mask_not_autostore = ~chunk["store_name"].fillna("").astype(str).str.startswith("AS_")

            # optional country code filter (single or multiple)
            if country_codes:
                if isinstance(country_codes, (list, tuple, set)):
                    mask_country = chunk["country_code"].isin(list(country_codes))
                else:
                    mask_country = chunk["country_code"] == country_codes
            else:
                mask_country = True

            sub = chunk[mask_valid_image & mask_has_description & mask_not_autostore & mask_country]
            if not sub.empty:
                if max_records and total_filtered + len(sub) > max_records:
                    remaining = max_records - total_filtered
                    sub = sub.head(remaining)
                    filtered_chunks.append(sub)
                    total_filtered += len(sub)
                    pbar.set_postfix(filtered_records=total_filtered)
                    pbar.update(1)
                    break
                
                filtered_chunks.append(sub)
                total_filtered += len(sub)
                
                if max_records and total_filtered >= max_records:
                    pbar.set_postfix(filtered_records=total_filtered)
                    pbar.update(1)
                    break
            
            pbar.set_postfix(filtered_records=total_filtered)
            pbar.update(1)

    if filtered_chunks:
        df = pd.concat(filtered_chunks, ignore_index=True)
    else:
        df = pd.DataFrame(columns=seen_columns or [])

    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest CSV from S3 and save filtered data as Parquet")
    parser.add_argument("--max-records", type=int, help="Maximum number of records to process")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help=f"Directory to save the Parquet file (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--output-file", type=str, 
                        help="Complete file path (e.g., 'data/custom.parquet'). Overrides --output-dir")
    parser.add_argument("--country-codes", type=str, nargs='*', help="Optional list of country_code filters (e.g., ES PT IT). If omitted, no country filter is applied.")
    
    args = parser.parse_args()
    ingest_s3_csv_to_parquet(
        max_records=args.max_records, 
        output_dir=args.output_dir,
        output_file=args.output_file,
        country_codes=args.country_codes
    )


if __name__ == "__main__":
    main()


