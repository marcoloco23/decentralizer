"""DuckDB storage layer replacing MongoDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from decentralizer.config import get_settings


def get_connection(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection, creating tables if needed."""
    if path is None:
        path = get_settings().duckdb_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    _create_tables(conn)
    return conn


def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blocks (
            chain_id INTEGER NOT NULL DEFAULT 1,
            number INTEGER NOT NULL,
            timestamp BIGINT NOT NULL,
            transaction_count INTEGER NOT NULL,
            PRIMARY KEY (chain_id, number)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            chain_id INTEGER NOT NULL DEFAULT 1,
            hash VARCHAR NOT NULL,
            block_number INTEGER NOT NULL,
            sender VARCHAR NOT NULL,
            receiver VARCHAR NOT NULL,
            value DOUBLE NOT NULL,
            timestamp BIGINT NOT NULL,
            gas BIGINT NOT NULL,
            gas_price BIGINT NOT NULL,
            max_fee_per_gas BIGINT,
            max_priority_fee_per_gas BIGINT,
            input_data VARCHAR DEFAULT '',
            tx_type INTEGER DEFAULT 0,
            PRIMARY KEY (chain_id, hash)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS addresses (
            chain_id INTEGER NOT NULL DEFAULT 1,
            address VARCHAR NOT NULL,
            page_rank DOUBLE DEFAULT 0.0,
            in_degree INTEGER DEFAULT 0,
            out_degree INTEGER DEFAULT 0,
            total_received DOUBLE DEFAULT 0.0,
            total_sent DOUBLE DEFAULT 0.0,
            tx_count INTEGER DEFAULT 0,
            PRIMARY KEY (chain_id, address)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS address_metrics (
            chain_id INTEGER NOT NULL DEFAULT 1,
            address VARCHAR NOT NULL,
            page_rank DOUBLE DEFAULT 0.0,
            weighted_page_rank DOUBLE DEFAULT 0.0,
            betweenness_centrality DOUBLE DEFAULT 0.0,
            clustering_coefficient DOUBLE DEFAULT 0.0,
            influence_score DOUBLE DEFAULT 0.0,
            community_id INTEGER DEFAULT -1,
            anomaly_score DOUBLE DEFAULT 0.0,
            PRIMARY KEY (chain_id, address)
        )
    """)


def insert_transactions(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Bulk insert transactions from a DataFrame. Returns rows inserted."""
    if df.empty:
        return 0
    conn.execute("""
        INSERT OR IGNORE INTO transactions
        SELECT * FROM df
    """)
    return len(df)


def insert_blocks(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.execute("INSERT OR IGNORE INTO blocks SELECT * FROM df")
    return len(df)


def get_transactions(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    financial_only: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Retrieve transactions as a DataFrame."""
    query = "SELECT * FROM transactions WHERE chain_id = ?"
    if financial_only:
        query += " AND value > 0"
    if limit:
        query += f" LIMIT {limit}"
    return conn.execute(query, [chain_id]).fetchdf()


def get_edge_dataframe(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    financial_only: bool = False,
) -> pd.DataFrame:
    """Get sender/receiver/value edge list for graph construction."""
    query = """
        SELECT sender, receiver, value, gas, gas_price, block_number, timestamp
        FROM transactions
        WHERE chain_id = ?
    """
    if financial_only:
        query += " AND value > 0"
    return conn.execute(query, [chain_id]).fetchdf()


def get_address_metrics(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
) -> pd.DataFrame:
    return conn.execute(
        "SELECT * FROM address_metrics WHERE chain_id = ?", [chain_id]
    ).fetchdf()


def upsert_address_metrics(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> None:
    """Upsert address metrics from a DataFrame."""
    if df.empty:
        return
    conn.execute("""
        INSERT OR REPLACE INTO address_metrics
        SELECT * FROM df
    """)


def get_transaction_count(conn: duckdb.DuckDBPyConnection, chain_id: int = 1) -> int:
    result = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE chain_id = ?", [chain_id]
    ).fetchone()
    return result[0] if result else 0


def get_unique_addresses(conn: duckdb.DuckDBPyConnection, chain_id: int = 1) -> list[str]:
    """Get all unique addresses (senders + receivers)."""
    result = conn.execute("""
        SELECT DISTINCT address FROM (
            SELECT sender AS address FROM transactions WHERE chain_id = ?
            UNION
            SELECT receiver AS address FROM transactions WHERE chain_id = ?
        )
    """, [chain_id, chain_id]).fetchdf()
    return result["address"].tolist()


def migrate_legacy_csvs(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Import legacy financial_transactions.csv and non_financial_transactions.csv."""
    settings = get_settings()
    counts = {}

    for label, csv_path in [
        ("financial", settings.financial_csv),
        ("non_financial", settings.non_financial_csv),
    ]:
        if not csv_path.exists():
            counts[label] = 0
            continue

        df = pd.read_csv(csv_path)

        # Normalize column names from legacy format
        rename_map = {
            "blockNumber": "block_number",
            "gasPrice": "gas_price",
            "inputData": "input_data",
        }
        df = df.rename(columns=rename_map)

        # Drop MongoDB _id if present
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Add missing columns for DuckDB schema
        df["chain_id"] = 1
        df["max_fee_per_gas"] = None
        df["max_priority_fee_per_gas"] = None
        df["tx_type"] = 0

        # Generate hash if missing (legacy data may not have it)
        if "hash" not in df.columns:
            df["hash"] = df.apply(
                lambda r: f"{r['sender']}_{r['receiver']}_{r['block_number']}_{r.name}",
                axis=1,
            )

        if "input_data" not in df.columns:
            df["input_data"] = ""

        # Ensure correct column order matching table schema
        cols = [
            "chain_id", "hash", "block_number", "sender", "receiver",
            "value", "timestamp", "gas", "gas_price",
            "max_fee_per_gas", "max_priority_fee_per_gas",
            "input_data", "tx_type",
        ]
        df = df[cols]

        conn.execute("INSERT OR IGNORE INTO transactions SELECT * FROM df")
        counts[label] = len(df)

    return counts
