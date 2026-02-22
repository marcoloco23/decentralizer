"""Per-token on-chain feature computation. SQL-heavy against DuckDB."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import duckdb

logger = logging.getLogger(__name__)

# Raw feature columns stored in token_features table
RAW_FEATURE_COLS = [
    "holder_count",
    "new_holders_7d",
    "holder_growth_rate",
    "top10_concentration",
    "gini_coefficient",
    "whale_transfer_count",
    "smart_money_inflow_pct",
    "smart_money_outflow_pct",
    "smart_money_net_flow",
    "smart_money_holder_pct",
    "daily_transfers",
    "daily_unique_senders",
    "daily_unique_receivers",
    "transfer_velocity",
    "dex_volume_usd",
    "dex_trade_count",
    "dex_unique_traders",
    "buy_sell_ratio",
]


def compute_token_features(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    token_address: str,
    symbol: str,
    as_of_date: date,
) -> dict:
    """Compute all on-chain features for one token on one date.

    Args:
        conn: DuckDB connection
        chain_id: Chain identifier
        token_address: Token contract address
        symbol: Numerai symbol
        as_of_date: Date to compute features for

    Returns:
        Dict of feature values keyed by column name
    """
    # Convert date to timestamp range (midnight to midnight UTC)
    day_start = int(pd.Timestamp(as_of_date).timestamp())
    day_end = day_start + 86400
    week_ago = day_start - 7 * 86400

    features: dict = {
        "symbol": symbol,
        "chain_id": chain_id,
        "date": as_of_date,
    }

    # --- Holder metrics ---
    holder_metrics = _compute_holder_metrics(conn, chain_id, token_address, day_end, week_ago)
    features.update(holder_metrics)

    # --- Concentration metrics ---
    conc_metrics = _compute_concentration_metrics(conn, chain_id, token_address, day_end)
    features.update(conc_metrics)

    # --- Smart money signals ---
    sm_metrics = _compute_smart_money_signals(conn, chain_id, token_address, day_start, day_end)
    features.update(sm_metrics)

    # --- Network activity ---
    activity_metrics = _compute_network_activity(conn, chain_id, token_address, day_start, day_end)
    features.update(activity_metrics)

    # --- Transfer velocity ---
    holder_count = features.get("holder_count", 0) or 0
    daily_transfers = features.get("daily_transfers", 0) or 0
    features["transfer_velocity"] = daily_transfers / max(holder_count, 1)

    # --- DEX trading signals ---
    dex_metrics = _compute_dex_signals(conn, chain_id, token_address, day_start, day_end)
    features.update(dex_metrics)

    return features


def compute_all_token_features(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Compute features for all mapped tokens on a given date.

    Args:
        conn: DuckDB connection
        chain_id: Chain identifier
        as_of_date: Date to compute for (defaults to today)

    Returns:
        DataFrame with one row per symbol, columns = feature names
    """
    from decentralizer.storage.database import get_token_mapping, upsert_token_features

    if as_of_date is None:
        as_of_date = date.today()

    mapping_df = get_token_mapping(conn, chain_id=chain_id)
    if mapping_df.empty:
        logger.warning(f"No token mappings for chain_id={chain_id}. Run `map-tokens` first.")
        return pd.DataFrame()

    logger.info(f"Computing features for {len(mapping_df)} tokens on {as_of_date}")

    rows = []
    for _, m in mapping_df.iterrows():
        try:
            features = compute_token_features(
                conn, chain_id, m["token_address"], m["symbol"], as_of_date,
            )
            rows.append(features)
        except Exception as e:
            logger.debug(f"Error computing features for {m['symbol']}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure column order matches table schema
    table_cols = ["symbol", "chain_id", "date"] + RAW_FEATURE_COLS
    for col in table_cols:
        if col not in df.columns:
            df[col] = None
    df = df[table_cols]

    upsert_token_features(conn, df)
    logger.info(f"Computed and stored features for {len(df)} tokens")

    return df


def _compute_holder_metrics(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    token_address: str,
    as_of_ts: int,
    week_ago_ts: int,
) -> dict:
    """Holder count, new holders in 7d, and growth rate."""
    result = conn.execute("""
        WITH balances AS (
            SELECT address, SUM(net) as balance FROM (
                SELECT to_address as address, SUM(COALESCE(value_decimal, 0)) as net
                FROM token_transfers
                WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                GROUP BY to_address
                UNION ALL
                SELECT from_address as address, -SUM(COALESCE(value_decimal, 0)) as net
                FROM token_transfers
                WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                GROUP BY from_address
            ) GROUP BY address
        ),
        current_holders AS (
            SELECT address FROM balances WHERE balance > 0
        ),
        -- Holders as of 7 days ago
        balances_7d AS (
            SELECT address, SUM(net) as balance FROM (
                SELECT to_address as address, SUM(COALESCE(value_decimal, 0)) as net
                FROM token_transfers
                WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                GROUP BY to_address
                UNION ALL
                SELECT from_address as address, -SUM(COALESCE(value_decimal, 0)) as net
                FROM token_transfers
                WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                GROUP BY from_address
            ) GROUP BY address
        ),
        old_holders AS (
            SELECT address FROM balances_7d WHERE balance > 0
        )
        SELECT
            (SELECT COUNT(*) FROM current_holders) as holder_count,
            (SELECT COUNT(*) FROM current_holders WHERE address NOT IN (SELECT address FROM old_holders)) as new_holders_7d,
            (SELECT COUNT(*) FROM old_holders) as holders_7d_ago
    """, [
        chain_id, token_address, as_of_ts,
        chain_id, token_address, as_of_ts,
        chain_id, token_address, week_ago_ts,
        chain_id, token_address, week_ago_ts,
    ]).fetchone()

    holder_count = result[0] if result else 0
    new_holders_7d = result[1] if result else 0
    holders_7d_ago = result[2] if result else 0

    growth_rate = 0.0
    if holders_7d_ago > 0:
        growth_rate = (holder_count - holders_7d_ago) / holders_7d_ago

    return {
        "holder_count": holder_count,
        "new_holders_7d": new_holders_7d,
        "holder_growth_rate": growth_rate,
    }


def _compute_concentration_metrics(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    token_address: str,
    as_of_ts: int,
) -> dict:
    """Top 10 concentration, Gini coefficient, whale transfer count."""
    # Get transfer volumes per address
    result = conn.execute("""
        WITH volumes AS (
            SELECT from_address as address, SUM(COALESCE(value_decimal, 0)) as volume
            FROM token_transfers
            WHERE chain_id = ? AND token_address = ? AND timestamp < ?
            GROUP BY from_address
            UNION ALL
            SELECT to_address as address, SUM(COALESCE(value_decimal, 0)) as volume
            FROM token_transfers
            WHERE chain_id = ? AND token_address = ? AND timestamp < ?
            GROUP BY to_address
        ),
        addr_volumes AS (
            SELECT address, SUM(volume) as total_volume
            FROM volumes GROUP BY address
            ORDER BY total_volume DESC
        ),
        total AS (
            SELECT SUM(total_volume) as grand_total FROM addr_volumes
        )
        SELECT
            -- Top 10 concentration
            (SELECT COALESCE(SUM(total_volume), 0) FROM (SELECT total_volume FROM addr_volumes LIMIT 10))
            / GREATEST((SELECT grand_total FROM total), 1) as top10_pct,
            -- Total addresses with volume
            (SELECT COUNT(*) FROM addr_volumes WHERE total_volume > 0) as n_addresses
    """, [
        chain_id, token_address, as_of_ts,
        chain_id, token_address, as_of_ts,
    ]).fetchone()

    top10_concentration = result[0] if result else 0

    # Gini coefficient via SQL
    gini = _compute_gini(conn, chain_id, token_address, as_of_ts)

    # Whale transfers (> 90th percentile value)
    whale_result = conn.execute("""
        WITH pct AS (
            SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY COALESCE(value_decimal, 0)) as p90
            FROM token_transfers
            WHERE chain_id = ? AND token_address = ? AND timestamp < ?
              AND COALESCE(value_decimal, 0) > 0
        )
        SELECT COUNT(*) FROM token_transfers tt, pct
        WHERE tt.chain_id = ? AND tt.token_address = ?
          AND tt.timestamp < ?
          AND COALESCE(tt.value_decimal, 0) > pct.p90
    """, [
        chain_id, token_address, as_of_ts,
        chain_id, token_address, as_of_ts,
    ]).fetchone()

    return {
        "top10_concentration": top10_concentration,
        "gini_coefficient": gini,
        "whale_transfer_count": whale_result[0] if whale_result else 0,
    }


def _compute_gini(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    token_address: str,
    as_of_ts: int,
) -> float:
    """Compute Gini coefficient of token holder balances."""
    result = conn.execute("""
        WITH balances AS (
            SELECT address, SUM(net) as balance FROM (
                SELECT to_address as address, SUM(COALESCE(value_decimal, 0)) as net
                FROM token_transfers
                WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                GROUP BY to_address
                UNION ALL
                SELECT from_address as address, -SUM(COALESCE(value_decimal, 0)) as net
                FROM token_transfers
                WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                GROUP BY from_address
            ) GROUP BY address
            HAVING balance > 0
        ),
        ranked AS (
            SELECT balance, ROW_NUMBER() OVER (ORDER BY balance) as rn,
                   COUNT(*) OVER () as n
            FROM balances
        )
        SELECT
            CASE WHEN MAX(n) <= 1 THEN 0.0
            ELSE 1.0 - 2.0 * SUM((n + 1 - rn) * balance) / (n * SUM(balance))
            END as gini
        FROM ranked
        GROUP BY n
    """, [
        chain_id, token_address, as_of_ts,
        chain_id, token_address, as_of_ts,
    ]).fetchone()

    return result[0] if result and result[0] is not None else 0.0


def _compute_smart_money_signals(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    token_address: str,
    day_start: int,
    day_end: int,
) -> dict:
    """Smart money inflow/outflow metrics for a token on a given day."""
    # Use smart_money_scores table to identify smart wallets (top 20% by composite)
    result = conn.execute("""
        WITH smart_wallets AS (
            SELECT address FROM smart_money_scores
            WHERE chain_id = ?
              AND composite_score >= (
                  SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY composite_score)
                  FROM smart_money_scores WHERE chain_id = ?
              )
        ),
        day_transfers AS (
            SELECT from_address, to_address, COALESCE(value_decimal, 0) as val
            FROM token_transfers
            WHERE chain_id = ? AND token_address = ?
              AND timestamp >= ? AND timestamp < ?
        ),
        inflow AS (
            SELECT COALESCE(SUM(val), 0) as sm_in
            FROM day_transfers dt
            WHERE dt.to_address IN (SELECT address FROM smart_wallets)
        ),
        outflow AS (
            SELECT COALESCE(SUM(val), 0) as sm_out
            FROM day_transfers dt
            WHERE dt.from_address IN (SELECT address FROM smart_wallets)
        ),
        total_in AS (
            SELECT COALESCE(SUM(val), 0) as total
            FROM day_transfers
        ),
        -- Smart money holders of this token
        sm_holders AS (
            SELECT COUNT(DISTINCT b.address) as cnt FROM (
                SELECT address, SUM(net) as balance FROM (
                    SELECT to_address as address, SUM(COALESCE(value_decimal, 0)) as net
                    FROM token_transfers
                    WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                    GROUP BY to_address
                    UNION ALL
                    SELECT from_address as address, -SUM(COALESCE(value_decimal, 0)) as net
                    FROM token_transfers
                    WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                    GROUP BY from_address
                ) GROUP BY address HAVING balance > 0
            ) b
            WHERE b.address IN (SELECT address FROM smart_wallets)
        ),
        all_holders AS (
            SELECT COUNT(DISTINCT address) as cnt FROM (
                SELECT address, SUM(net) as balance FROM (
                    SELECT to_address as address, SUM(COALESCE(value_decimal, 0)) as net
                    FROM token_transfers
                    WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                    GROUP BY to_address
                    UNION ALL
                    SELECT from_address as address, -SUM(COALESCE(value_decimal, 0)) as net
                    FROM token_transfers
                    WHERE chain_id = ? AND token_address = ? AND timestamp < ?
                    GROUP BY from_address
                ) GROUP BY address HAVING balance > 0
            )
        )
        SELECT
            (SELECT sm_in FROM inflow),
            (SELECT sm_out FROM outflow),
            (SELECT total FROM total_in),
            (SELECT cnt FROM sm_holders),
            (SELECT cnt FROM all_holders)
    """, [
        chain_id, chain_id,
        chain_id, token_address, day_start, day_end,
        chain_id, token_address, day_end,
        chain_id, token_address, day_end,
        chain_id, token_address, day_end,
        chain_id, token_address, day_end,
    ]).fetchone()

    sm_in = result[0] if result else 0
    sm_out = result[1] if result else 0
    total_vol = result[2] if result else 0
    sm_holder_count = result[3] if result else 0
    total_holder_count = result[4] if result else 0

    return {
        "smart_money_inflow_pct": sm_in / max(total_vol, 1),
        "smart_money_outflow_pct": sm_out / max(total_vol, 1),
        "smart_money_net_flow": sm_in - sm_out,
        "smart_money_holder_pct": sm_holder_count / max(total_holder_count, 1),
    }


def _compute_network_activity(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    token_address: str,
    day_start: int,
    day_end: int,
) -> dict:
    """Daily transfer count, unique senders/receivers."""
    result = conn.execute("""
        SELECT
            COUNT(*) as daily_transfers,
            COUNT(DISTINCT from_address) as daily_unique_senders,
            COUNT(DISTINCT to_address) as daily_unique_receivers
        FROM token_transfers
        WHERE chain_id = ? AND token_address = ?
          AND timestamp >= ? AND timestamp < ?
    """, [chain_id, token_address, day_start, day_end]).fetchone()

    return {
        "daily_transfers": result[0] if result else 0,
        "daily_unique_senders": result[1] if result else 0,
        "daily_unique_receivers": result[2] if result else 0,
    }


def _compute_dex_signals(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
    token_address: str,
    day_start: int,
    day_end: int,
) -> dict:
    """DEX volume, trade count, unique traders, buy/sell ratio."""
    result = conn.execute("""
        WITH day_trades AS (
            SELECT *
            FROM dex_trades
            WHERE chain_id = ? AND timestamp >= ? AND timestamp < ?
              AND (token_in = ? OR token_out = ?)
        )
        SELECT
            COALESCE(SUM(amount_usd), 0) as dex_volume_usd,
            COUNT(*) as dex_trade_count,
            COUNT(DISTINCT trader) as dex_unique_traders,
            -- Buy = token_out (someone received this token), Sell = token_in
            SUM(CASE WHEN token_out = ? THEN 1 ELSE 0 END) as buys,
            SUM(CASE WHEN token_in = ? THEN 1 ELSE 0 END) as sells
        FROM day_trades
    """, [
        chain_id, day_start, day_end, token_address, token_address,
        token_address, token_address,
    ]).fetchone()

    buys = result[3] if result else 0
    sells = result[4] if result else 0

    return {
        "dex_volume_usd": result[0] if result else 0,
        "dex_trade_count": result[1] if result else 0,
        "dex_unique_traders": result[2] if result else 0,
        "buy_sell_ratio": buys / max(sells, 1),
    }
