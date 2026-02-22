"""Composite smart money scoring: PageRank + P&L + early entry + concentration."""

from __future__ import annotations

import time

import pandas as pd
import duckdb


def compute_smart_money_scores(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    top_k: int = 1000,
) -> pd.DataFrame:
    """Compute composite smart money score.

    Components (each normalized 0-1):
    - 0.3 x PageRank: network influence from address_metrics
    - 0.4 x P&L: total realized P&L from wallet_pnl
    - 0.2 x Early entry: how often wallet bought tokens in first 10% of holder history
    - 0.1 x Concentration: inverse HHI of token portfolio (diversified = higher)
    """
    # 1. PageRank scores
    pr_df = conn.execute("""
        SELECT address, page_rank
        FROM address_metrics
        WHERE chain_id = ? AND page_rank > 0
    """, [chain_id]).fetchdf()

    # 2. P&L scores
    pnl_df = conn.execute("""
        SELECT address, SUM(total_pnl) as total_pnl
        FROM wallet_pnl WHERE chain_id = ?
        GROUP BY address
    """, [chain_id]).fetchdf()

    # 3. Early entry scores
    early_df = _compute_early_entry_scores(conn, chain_id)

    # 4. Concentration (diversification) scores
    conc_df = _compute_concentration_scores(conn, chain_id)

    # Merge all components
    if pr_df.empty and pnl_df.empty:
        return pd.DataFrame()

    merged = pr_df.copy() if not pr_df.empty else pd.DataFrame(columns=["address", "page_rank"])
    if not pnl_df.empty:
        merged = merged.merge(pnl_df, on="address", how="outer")
    if not early_df.empty:
        merged = merged.merge(early_df, on="address", how="outer")
    if not conc_df.empty:
        merged = merged.merge(conc_df, on="address", how="outer")

    merged = merged.fillna(0)

    # Min-max normalize each component
    merged["page_rank_score"] = _normalize(merged.get("page_rank", pd.Series(dtype=float)))
    merged["pnl_score"] = _normalize(merged.get("total_pnl", pd.Series(dtype=float)))
    merged["early_entry_score"] = _normalize(merged.get("early_entry_raw", pd.Series(dtype=float)))
    merged["concentration_score"] = _normalize(merged.get("concentration_raw", pd.Series(dtype=float)))

    # Composite score
    merged["composite_score"] = (
        0.3 * merged["page_rank_score"]
        + 0.4 * merged["pnl_score"]
        + 0.2 * merged["early_entry_score"]
        + 0.1 * merged["concentration_score"]
    )

    merged["rank"] = merged["composite_score"].rank(ascending=False, method="min").astype(int)
    merged["chain_id"] = chain_id
    merged["last_updated"] = int(time.time())

    result = merged.nlargest(top_k, "composite_score")

    return result[
        [
            "chain_id",
            "address",
            "page_rank_score",
            "pnl_score",
            "early_entry_score",
            "concentration_score",
            "composite_score",
            "rank",
            "last_updated",
        ]
    ].reset_index(drop=True)


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to 0-1."""
    if series.empty:
        return series
    mn, mx = series.min(), series.max()
    if mx > mn:
        return (series - mn) / (mx - mn)
    return pd.Series(0.0, index=series.index)


def _compute_early_entry_scores(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
) -> pd.DataFrame:
    """Score wallets by how often they bought tokens in the first 10% of holder history.

    For each token, find the earliest 10% of transfers. Wallets appearing as
    receivers in that early window get early_entry credit.
    """
    result = conn.execute("""
        WITH token_ranges AS (
            SELECT
                token_address,
                MIN(block_number) as min_block,
                MAX(block_number) as max_block,
                COUNT(*) as total_transfers
            FROM token_transfers
            WHERE chain_id = ?
            GROUP BY token_address
            HAVING total_transfers >= 10
        ),
        early_window AS (
            SELECT
                token_address,
                min_block,
                min_block + CAST((max_block - min_block) * 0.1 AS INTEGER) as early_cutoff
            FROM token_ranges
        ),
        early_buyers AS (
            SELECT tt.to_address as address, COUNT(DISTINCT tt.token_address) as early_tokens
            FROM token_transfers tt
            JOIN early_window ew
              ON tt.token_address = ew.token_address
              AND tt.block_number <= ew.early_cutoff
            WHERE tt.chain_id = ?
            GROUP BY tt.to_address
        )
        SELECT address, early_tokens as early_entry_raw
        FROM early_buyers
    """, [chain_id, chain_id]).fetchdf()

    return result if not result.empty else pd.DataFrame(columns=["address", "early_entry_raw"])


def _compute_concentration_scores(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int,
) -> pd.DataFrame:
    """Score wallets by portfolio diversification (inverse HHI).

    HHI = sum of squared portfolio weights. Lower HHI = more diversified.
    Score = 1 - HHI (so diversified wallets score higher).
    """
    result = conn.execute("""
        WITH holdings AS (
            SELECT
                address,
                token_address,
                quantity * COALESCE(
                    (SELECT price_usd FROM token_prices tp
                     WHERE tp.chain_id = wp.chain_id AND tp.token_address = wp.token_address
                     ORDER BY date DESC LIMIT 1), 0
                ) as value_usd
            FROM wallet_pnl wp
            WHERE chain_id = ? AND quantity > 0
        ),
        portfolio_total AS (
            SELECT address, SUM(value_usd) as total_value
            FROM holdings
            GROUP BY address
            HAVING total_value > 0
        ),
        weights AS (
            SELECT
                h.address,
                (h.value_usd / pt.total_value) as weight
            FROM holdings h
            JOIN portfolio_total pt ON h.address = pt.address
            WHERE pt.total_value > 0
        ),
        hhi AS (
            SELECT address, SUM(weight * weight) as hhi_value
            FROM weights
            GROUP BY address
        )
        SELECT address, 1.0 - hhi_value as concentration_raw
        FROM hhi
    """, [chain_id]).fetchdf()

    return result if not result.empty else pd.DataFrame(columns=["address", "concentration_raw"])
