"""
Data ingestion utilities for S3 CSV processing
"""

from .ingest_s3_csv import ingest_s3_csv_to_parquet

__all__ = ["ingest_s3_csv_to_parquet"]
