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
    # --- Token transfers ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_transfers (
            chain_id INTEGER NOT NULL DEFAULT 1,
            tx_hash VARCHAR NOT NULL,
            log_index INTEGER NOT NULL,
            block_number INTEGER NOT NULL,
            token_address VARCHAR NOT NULL,
            from_address VARCHAR NOT NULL,
            to_address VARCHAR NOT NULL,
            value_raw VARCHAR NOT NULL,
            value_decimal DOUBLE,
            timestamp BIGINT NOT NULL DEFAULT 0,
            PRIMARY KEY (chain_id, tx_hash, log_index)
        )
    """)
    # --- Token metadata ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_metadata (
            chain_id INTEGER NOT NULL DEFAULT 1,
            address VARCHAR NOT NULL,
            symbol VARCHAR DEFAULT '',
            name VARCHAR DEFAULT '',
            decimals INTEGER DEFAULT 18,
            PRIMARY KEY (chain_id, address)
        )
    """)
    # --- Token prices (daily) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_prices (
            chain_id INTEGER NOT NULL DEFAULT 1,
            token_address VARCHAR NOT NULL,
            date DATE NOT NULL,
            price_usd DOUBLE NOT NULL,
            source VARCHAR DEFAULT 'defillama',
            PRIMARY KEY (chain_id, token_address, date)
        )
    """)
    # --- DEX trades ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dex_trades (
            chain_id INTEGER NOT NULL DEFAULT 1,
            tx_hash VARCHAR NOT NULL,
            log_index INTEGER NOT NULL,
            block_number INTEGER NOT NULL,
            timestamp BIGINT NOT NULL,
            dex VARCHAR NOT NULL,
            trader VARCHAR NOT NULL,
            token_in VARCHAR NOT NULL,
            token_out VARCHAR NOT NULL,
            amount_in DOUBLE NOT NULL,
            amount_out DOUBLE NOT NULL,
            amount_usd DOUBLE,
            PRIMARY KEY (chain_id, tx_hash, log_index)
        )
    """)
    # --- Wallet P&L ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wallet_pnl (
            chain_id INTEGER NOT NULL DEFAULT 1,
            address VARCHAR NOT NULL,
            token_address VARCHAR NOT NULL,
            cost_basis DOUBLE DEFAULT 0.0,
            quantity DOUBLE DEFAULT 0.0,
            realized_pnl DOUBLE DEFAULT 0.0,
            unrealized_pnl DOUBLE DEFAULT 0.0,
            total_pnl DOUBLE DEFAULT 0.0,
            last_updated BIGINT NOT NULL DEFAULT 0,
            PRIMARY KEY (chain_id, address, token_address)
        )
    """)
    # --- Smart money scores ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS smart_money_scores (
            chain_id INTEGER NOT NULL DEFAULT 1,
            address VARCHAR NOT NULL,
            page_rank_score DOUBLE DEFAULT 0.0,
            pnl_score DOUBLE DEFAULT 0.0,
            early_entry_score DOUBLE DEFAULT 0.0,
            concentration_score DOUBLE DEFAULT 0.0,
            composite_score DOUBLE DEFAULT 0.0,
            rank INTEGER,
            last_updated BIGINT NOT NULL DEFAULT 0,
            PRIMARY KEY (chain_id, address)
        )
    """)

    # --- Token mapping (Numerai symbol → on-chain address) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_mapping (
            symbol VARCHAR NOT NULL,
            ucid INTEGER,
            coingecko_id VARCHAR,
            chain_id INTEGER NOT NULL,
            token_address VARCHAR NOT NULL,
            is_native BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (symbol, chain_id)
        )
    """)
    # --- Token features (per-token per-date on-chain features) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_features (
            symbol VARCHAR NOT NULL,
            chain_id INTEGER NOT NULL,
            date DATE NOT NULL,
            holder_count INTEGER,
            new_holders_7d INTEGER,
            holder_growth_rate DOUBLE,
            top10_concentration DOUBLE,
            gini_coefficient DOUBLE,
            whale_transfer_count INTEGER,
            smart_money_inflow_pct DOUBLE,
            smart_money_outflow_pct DOUBLE,
            smart_money_net_flow DOUBLE,
            smart_money_holder_pct DOUBLE,
            daily_transfers INTEGER,
            daily_unique_senders INTEGER,
            daily_unique_receivers INTEGER,
            transfer_velocity DOUBLE,
            dex_volume_usd DOUBLE,
            dex_trade_count INTEGER,
            dex_unique_traders INTEGER,
            buy_sell_ratio DOUBLE,
            PRIMARY KEY (symbol, chain_id, date)
        )
    """)

    # Indexes for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_sender ON transactions(chain_id, sender)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_receiver ON transactions(chain_id, receiver)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_value ON transactions(chain_id, value)")
    # Token transfer indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tt_from ON token_transfers(chain_id, from_address)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tt_to ON token_transfers(chain_id, to_address)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tt_token ON token_transfers(chain_id, token_address)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tt_block ON token_transfers(chain_id, block_number)")
    # DEX trade indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dex_trader ON dex_trades(chain_id, trader)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dex_token_in ON dex_trades(chain_id, token_in)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dex_token_out ON dex_trades(chain_id, token_out)")
    # Smart money index
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sm_composite ON smart_money_scores(chain_id, composite_score)")
    # Token mapping indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_symbol ON token_mapping(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_address ON token_mapping(chain_id, token_address)")
    # Token features indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tf_date ON token_features(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tf_symbol_date ON token_features(symbol, date)")


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


## Token transfers CRUD ##

def insert_token_transfers(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.execute("INSERT OR IGNORE INTO token_transfers SELECT * FROM df")
    return len(df)


def get_token_transfers_for_address(
    conn: duckdb.DuckDBPyConnection,
    address: str,
    chain_id: int = 1,
    limit: int = 1000,
) -> pd.DataFrame:
    return conn.execute("""
        SELECT * FROM token_transfers
        WHERE chain_id = ? AND (from_address = ? OR to_address = ?)
        ORDER BY block_number DESC
        LIMIT ?
    """, [chain_id, address, address, limit]).fetchdf()


def get_token_transfer_count(conn: duckdb.DuckDBPyConnection, chain_id: int = 1) -> int:
    result = conn.execute(
        "SELECT COUNT(*) FROM token_transfers WHERE chain_id = ?", [chain_id]
    ).fetchone()
    return result[0] if result else 0


## Token metadata CRUD ##

def insert_token_metadata(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.execute("INSERT OR IGNORE INTO token_metadata SELECT * FROM df")
    return len(df)


def upsert_token_metadata(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn.execute("INSERT OR REPLACE INTO token_metadata SELECT * FROM df")


def get_token_metadata(
    conn: duckdb.DuckDBPyConnection, chain_id: int = 1
) -> pd.DataFrame:
    return conn.execute(
        "SELECT * FROM token_metadata WHERE chain_id = ?", [chain_id]
    ).fetchdf()


## Token prices CRUD ##

def insert_token_prices(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.execute("INSERT OR IGNORE INTO token_prices SELECT * FROM df")
    return len(df)


def get_latest_price(
    conn: duckdb.DuckDBPyConnection,
    token_address: str,
    chain_id: int = 1,
) -> float | None:
    result = conn.execute("""
        SELECT price_usd FROM token_prices
        WHERE chain_id = ? AND token_address = ?
        ORDER BY date DESC LIMIT 1
    """, [chain_id, token_address]).fetchone()
    return result[0] if result else None


## DEX trades CRUD ##

def insert_dex_trades(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.execute("INSERT OR IGNORE INTO dex_trades SELECT * FROM df")
    return len(df)


def get_dex_trades_for_address(
    conn: duckdb.DuckDBPyConnection,
    address: str,
    chain_id: int = 1,
    limit: int = 1000,
) -> pd.DataFrame:
    return conn.execute("""
        SELECT * FROM dex_trades
        WHERE chain_id = ? AND trader = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, [chain_id, address, limit]).fetchdf()


def get_dex_trade_count(conn: duckdb.DuckDBPyConnection, chain_id: int = 1) -> int:
    result = conn.execute(
        "SELECT COUNT(*) FROM dex_trades WHERE chain_id = ?", [chain_id]
    ).fetchone()
    return result[0] if result else 0


## Wallet P&L CRUD ##

def upsert_wallet_pnl(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn.execute("INSERT OR REPLACE INTO wallet_pnl SELECT * FROM df")


def get_wallet_pnl(
    conn: duckdb.DuckDBPyConnection,
    address: str,
    chain_id: int = 1,
) -> pd.DataFrame:
    return conn.execute("""
        SELECT * FROM wallet_pnl
        WHERE chain_id = ? AND address = ?
        ORDER BY total_pnl DESC
    """, [chain_id, address]).fetchdf()


def get_top_pnl_wallets(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    top_k: int = 100,
) -> pd.DataFrame:
    return conn.execute("""
        SELECT address, SUM(total_pnl) as total_pnl,
               SUM(realized_pnl) as realized_pnl,
               SUM(unrealized_pnl) as unrealized_pnl,
               COUNT(DISTINCT token_address) as tokens_held
        FROM wallet_pnl
        WHERE chain_id = ?
        GROUP BY address
        ORDER BY total_pnl DESC
        LIMIT ?
    """, [chain_id, top_k]).fetchdf()


## Smart money scores CRUD ##

def upsert_smart_money_scores(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn.execute("INSERT OR REPLACE INTO smart_money_scores SELECT * FROM df")


def get_smart_money_leaderboard(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    top_k: int = 100,
) -> pd.DataFrame:
    return conn.execute("""
        SELECT * FROM smart_money_scores
        WHERE chain_id = ?
        ORDER BY composite_score DESC
        LIMIT ?
    """, [chain_id, top_k]).fetchdf()


def get_smart_money_score(
    conn: duckdb.DuckDBPyConnection,
    address: str,
    chain_id: int = 1,
) -> dict | None:
    result = conn.execute("""
        SELECT * FROM smart_money_scores
        WHERE chain_id = ? AND address = ?
    """, [chain_id, address]).fetchdf()
    if result.empty:
        return None
    return result.iloc[0].to_dict()


## Token mapping CRUD ##

def upsert_token_mapping(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn.execute("INSERT OR REPLACE INTO token_mapping SELECT * FROM df")


def get_token_mapping(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int | None = None,
) -> pd.DataFrame:
    if chain_id is not None:
        return conn.execute(
            "SELECT * FROM token_mapping WHERE chain_id = ?", [chain_id]
        ).fetchdf()
    return conn.execute("SELECT * FROM token_mapping").fetchdf()


## Token features CRUD ##

def upsert_token_features(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn.execute("INSERT OR REPLACE INTO token_features SELECT * FROM df")


def get_token_features(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    date: str | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM token_features WHERE chain_id = ?"
    params: list = [chain_id]
    if date:
        query += " AND date = ?"
        params.append(date)
    return conn.execute(query, params).fetchdf()


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
