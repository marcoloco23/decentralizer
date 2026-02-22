"""DuckDB storage layer. All heavy lifting pushed to SQL."""

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
    # Indexes for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_sender ON transactions(chain_id, sender)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_receiver ON transactions(chain_id, receiver)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_value ON transactions(chain_id, value)")


def insert_transactions(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.execute("INSERT OR IGNORE INTO transactions SELECT * FROM df")
    return len(df)


def insert_blocks(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.execute("INSERT OR IGNORE INTO blocks SELECT * FROM df")
    return len(df)


def get_transactions_for_address(
    conn: duckdb.DuckDBPyConnection,
    address: str,
    chain_id: int = 1,
    limit: int = 1000,
) -> pd.DataFrame:
    """Get transactions involving a specific address (uses index)."""
    return conn.execute("""
        SELECT hash, block_number, sender, receiver, value, timestamp, gas, gas_price
        FROM transactions
        WHERE chain_id = ? AND (sender = ? OR receiver = ?)
        ORDER BY timestamp DESC
        LIMIT ?
    """, [chain_id, address, address, limit]).fetchdf()


def get_address_summary(
    conn: duckdb.DuckDBPyConnection,
    address: str,
    chain_id: int = 1,
) -> dict:
    """Get summary stats for an address via SQL aggregation."""
    result = conn.execute("""
        SELECT
            COUNT(*) as tx_count,
            SUM(CASE WHEN sender = ? THEN 1 ELSE 0 END) as sent_count,
            SUM(CASE WHEN receiver = ? THEN 1 ELSE 0 END) as recv_count,
            SUM(value) as total_value,
            SUM(CASE WHEN sender = ? THEN value ELSE 0 END) as total_sent,
            SUM(CASE WHEN receiver = ? THEN value ELSE 0 END) as total_received
        FROM transactions
        WHERE chain_id = ? AND (sender = ? OR receiver = ?)
    """, [address, address, address, address, chain_id, address, address]).fetchone()
    if not result or result[0] == 0:
        return {}
    return {
        "tx_count": result[0], "sent_count": result[1], "recv_count": result[2],
        "total_value": result[3], "total_sent": result[4], "total_received": result[5],
    }


def get_overview_stats(conn: duckdb.DuckDBPyConnection, chain_id: int = 1) -> dict:
    """Get dataset overview stats via SQL (no full table scan in Python)."""
    result = conn.execute("""
        SELECT
            COUNT(*) as tx_count,
            COUNT(DISTINCT sender) as unique_senders,
            COUNT(DISTINCT receiver) as unique_receivers
        FROM transactions WHERE chain_id = ?
    """, [chain_id]).fetchone()
    return {"tx_count": result[0], "unique_senders": result[1], "unique_receivers": result[2]}


def get_edge_dataframe(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    financial_only: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Get edge list for visualization. Use limit for large datasets."""
    query = """
        SELECT sender, receiver, value, gas, gas_price, block_number, timestamp
        FROM transactions WHERE chain_id = ?
    """
    params = [chain_id]
    if financial_only:
        query += " AND value > 0"
    if limit:
        query += f" LIMIT {limit}"
    return conn.execute(query, params).fetchdf()


def get_address_metrics(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    address: str | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM address_metrics WHERE chain_id = ?"
    params: list = [chain_id]
    if address:
        query += " AND address = ?"
        params.append(address)
    return conn.execute(query, params).fetchdf()


def upsert_address_metrics(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn.execute("INSERT OR REPLACE INTO address_metrics SELECT * FROM df")


def get_transaction_count(conn: duckdb.DuckDBPyConnection, chain_id: int = 1) -> int:
    result = conn.execute("SELECT COUNT(*) FROM transactions WHERE chain_id = ?", [chain_id]).fetchone()
    return result[0] if result else 0


def get_unique_addresses(conn: duckdb.DuckDBPyConnection, chain_id: int = 1) -> list[str]:
    result = conn.execute("""
        SELECT DISTINCT address FROM (
            SELECT sender AS address FROM transactions WHERE chain_id = ?
            UNION
            SELECT receiver AS address FROM transactions WHERE chain_id = ?
        )
    """, [chain_id, chain_id]).fetchdf()
    return result["address"].tolist()


def migrate_legacy_csvs(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Import legacy CSVs into DuckDB."""
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
        rename_map = {"blockNumber": "block_number", "gasPrice": "gas_price", "inputData": "input_data"}
        df = df.rename(columns=rename_map)

        for col in ["_id", "Unnamed: 0"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df["chain_id"] = 1
        df["max_fee_per_gas"] = None
        df["max_priority_fee_per_gas"] = None
        df["tx_type"] = 0

        if "hash" not in df.columns:
            df["hash"] = [f"{r['sender']}_{r['receiver']}_{r['block_number']}_{i}" for i, r in df.iterrows()]

        if "input_data" not in df.columns:
            df["input_data"] = ""

        cols = [
            "chain_id", "hash", "block_number", "sender", "receiver",
            "value", "timestamp", "gas", "gas_price",
            "max_fee_per_gas", "max_priority_fee_per_gas", "input_data", "tx_type",
        ]
        df = df[cols]
        conn.execute("INSERT OR IGNORE INTO transactions SELECT * FROM df")
        counts[label] = len(df)

    return counts
