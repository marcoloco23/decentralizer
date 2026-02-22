"""Parquet export/import for columnar data exchange."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def export_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    output_path: Path,
    chain_id: int | None = None,
) -> Path:
    """Export a DuckDB table to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = f"SELECT * FROM {table}"
    if chain_id is not None:
        query += f" WHERE chain_id = {chain_id}"
    conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")
    return output_path


def import_from_parquet(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    input_path: Path,
) -> int:
    """Import a Parquet file into a DuckDB table."""
    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {input_path}")
    df = pd.read_parquet(input_path)
    conn.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM df")
    return len(df)
