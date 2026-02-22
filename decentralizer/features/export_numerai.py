"""Export quantile-normalized on-chain features for Numerai inference pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import duckdb

from decentralizer.features.on_chain import RAW_FEATURE_COLS

logger = logging.getLogger(__name__)


def export_inference_features(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    output_path: str | Path = "data/inference_features.parquet",
) -> Path:
    """Export quantile-normalized on-chain features matching Numerai schema.

    Reads raw features from token_features table, applies rolling windows,
    quantile-normalizes, and writes Parquet.

    Output schema:
    - 'symbol' column for merge key
    - All feature columns prefixed with 'feature_'
    - Values: float32, quantile-normalized [0, 1]

    Args:
        conn: DuckDB connection
        chain_id: Chain to export features for
        output_path: Output Parquet path

    Returns:
        Path to written file
    """
    from decentralizer.features.normalize import quantile_normalize, add_rolling_features
    from decentralizer.storage.database import get_token_features

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all historical features for this chain
    features_df = get_token_features(conn, chain_id=chain_id)

    if features_df.empty:
        logger.warning("No token features found. Run `compute-features` first.")
        # Write empty parquet with correct schema
        empty = pd.DataFrame(columns=["symbol"])
        empty.to_parquet(output_path, index=False)
        return output_path

    # Add rolling window features (20d, 60d averages + acceleration)
    features_df = add_rolling_features(features_df, RAW_FEATURE_COLS, windows=[20, 60])

    # Determine all columns to normalize:
    # raw features + rolling averages + acceleration ratios
    all_feature_cols = list(RAW_FEATURE_COLS)
    for col in RAW_FEATURE_COLS:
        all_feature_cols.append(f"{col}_avg_20d")
        all_feature_cols.append(f"{col}_avg_60d")
        all_feature_cols.append(f"{col}_accel")

    # Only include columns that exist
    all_feature_cols = [c for c in all_feature_cols if c in features_df.columns]

    # Quantile normalize (cross-sectional per date)
    features_df = quantile_normalize(features_df, all_feature_cols)

    # Take latest date only for inference
    latest_date = features_df["date"].max()
    latest_df = features_df[features_df["date"] == latest_date].copy()
    logger.info(f"Exporting features for {len(latest_df)} symbols as of {latest_date}")

    # Select output columns: symbol + all feature_* columns
    feature_output_cols = [c for c in latest_df.columns if c.startswith("feature_")]
    output_df = latest_df[["symbol"] + feature_output_cols].copy()

    # Convert to float32 for Numerai compatibility
    for col in feature_output_cols:
        output_df[col] = output_df[col].astype("float32")

    output_df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(output_df)} rows × {len(feature_output_cols)} features to {output_path}")

    return output_path


def upload_to_s3(
    parquet_path: str | Path,
    bucket: str = "eywa-ml-data",
    key: str = "inference_features/crypto/latest.parquet",
) -> str:
    """Upload feature parquet to S3 inference store.

    Args:
        parquet_path: Local path to parquet file
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        S3 URI of uploaded file
    """
    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(str(parquet_path), bucket, key)

    s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"Uploaded {parquet_path} to {s3_uri}")
    return s3_uri
