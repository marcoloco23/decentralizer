"""Export feature datasets as Parquet/CSV for Numerai pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import duckdb


def export_feature_dataset(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    output_path: str | Path = "data/features.parquet",
) -> Path:
    """Export a comprehensive feature dataset with one row per address.

    Features included:
    - Graph: page_rank, weighted_page_rank, betweenness, clustering_coeff,
             community_id, influence_score, in_degree, out_degree
    - P&L: realized_pnl, unrealized_pnl, total_pnl, cost_basis, num_tokens_held
    - Smart money: composite_score, pnl_score, early_entry_score, concentration_score
    - Activity: tx_count, token_transfer_count, dex_trade_count, unique_tokens_traded
    - Community: community_size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the feature dataset via SQL joins
    features_df = conn.execute("""
        WITH
        -- Graph features from address_metrics
        graph_features AS (
            SELECT
                address,
                page_rank,
                weighted_page_rank,
                betweenness_centrality,
                clustering_coefficient,
                influence_score,
                community_id
            FROM address_metrics
            WHERE chain_id = ?
        ),
        -- Degree counts from transactions
        degree_counts AS (
            SELECT address,
                   SUM(out_deg) as out_degree,
                   SUM(in_deg) as in_degree,
                   SUM(out_deg + in_deg) as tx_count
            FROM (
                SELECT sender as address, COUNT(*) as out_deg, 0 as in_deg
                FROM transactions WHERE chain_id = ? GROUP BY sender
                UNION ALL
                SELECT receiver as address, 0 as out_deg, COUNT(*) as in_deg
                FROM transactions WHERE chain_id = ? GROUP BY receiver
            )
            GROUP BY address
        ),
        -- P&L features from wallet_pnl
        pnl_features AS (
            SELECT
                address,
                SUM(realized_pnl) as realized_pnl,
                SUM(unrealized_pnl) as unrealized_pnl,
                SUM(total_pnl) as total_pnl,
                SUM(cost_basis) as total_cost_basis,
                COUNT(DISTINCT token_address) as num_tokens_held
            FROM wallet_pnl
            WHERE chain_id = ?
            GROUP BY address
        ),
        -- Smart money scores
        sm_features AS (
            SELECT
                address,
                page_rank_score as sm_page_rank_score,
                pnl_score as sm_pnl_score,
                early_entry_score as sm_early_entry_score,
                concentration_score as sm_concentration_score,
                composite_score as sm_composite_score,
                rank as sm_rank
            FROM smart_money_scores
            WHERE chain_id = ?
        ),
        -- Token transfer activity
        tt_activity AS (
            SELECT address, SUM(cnt) as token_transfer_count, COUNT(DISTINCT token) as unique_tokens_traded
            FROM (
                SELECT from_address as address, COUNT(*) as cnt, token_address as token
                FROM token_transfers WHERE chain_id = ? GROUP BY from_address, token_address
                UNION ALL
                SELECT to_address as address, COUNT(*) as cnt, token_address as token
                FROM token_transfers WHERE chain_id = ? GROUP BY to_address, token_address
            )
            GROUP BY address
        ),
        -- DEX trade activity
        dex_activity AS (
            SELECT trader as address, COUNT(*) as dex_trade_count
            FROM dex_trades WHERE chain_id = ?
            GROUP BY trader
        ),
        -- Community sizes
        comm_sizes AS (
            SELECT community_id, COUNT(*) as community_size
            FROM address_metrics
            WHERE chain_id = ? AND community_id >= 0
            GROUP BY community_id
        )
        SELECT
            gf.address,
            -- Graph features
            COALESCE(gf.page_rank, 0) as page_rank,
            COALESCE(gf.weighted_page_rank, 0) as weighted_page_rank,
            COALESCE(gf.betweenness_centrality, 0) as betweenness_centrality,
            COALESCE(gf.clustering_coefficient, 0) as clustering_coefficient,
            COALESCE(gf.influence_score, 0) as influence_score,
            COALESCE(gf.community_id, -1) as community_id,
            COALESCE(dc.in_degree, 0) as in_degree,
            COALESCE(dc.out_degree, 0) as out_degree,
            COALESCE(dc.tx_count, 0) as tx_count,
            -- P&L features
            COALESCE(pf.realized_pnl, 0) as realized_pnl,
            COALESCE(pf.unrealized_pnl, 0) as unrealized_pnl,
            COALESCE(pf.total_pnl, 0) as total_pnl,
            COALESCE(pf.total_cost_basis, 0) as total_cost_basis,
            COALESCE(pf.num_tokens_held, 0) as num_tokens_held,
            -- Smart money features
            COALESCE(sm.sm_page_rank_score, 0) as sm_page_rank_score,
            COALESCE(sm.sm_pnl_score, 0) as sm_pnl_score,
            COALESCE(sm.sm_early_entry_score, 0) as sm_early_entry_score,
            COALESCE(sm.sm_concentration_score, 0) as sm_concentration_score,
            COALESCE(sm.sm_composite_score, 0) as sm_composite_score,
            COALESCE(sm.sm_rank, 0) as sm_rank,
            -- Activity features
            COALESCE(ta.token_transfer_count, 0) as token_transfer_count,
            COALESCE(ta.unique_tokens_traded, 0) as unique_tokens_traded,
            COALESCE(da.dex_trade_count, 0) as dex_trade_count,
            -- Community features
            COALESCE(cs.community_size, 0) as community_size
        FROM graph_features gf
        LEFT JOIN degree_counts dc ON gf.address = dc.address
        LEFT JOIN pnl_features pf ON gf.address = pf.address
        LEFT JOIN sm_features sm ON gf.address = sm.address
        LEFT JOIN tt_activity ta ON gf.address = ta.address
        LEFT JOIN dex_activity da ON gf.address = da.address
        LEFT JOIN comm_sizes cs ON gf.community_id = cs.community_id
        ORDER BY COALESCE(sm.sm_composite_score, 0) DESC
    """, [chain_id, chain_id, chain_id, chain_id, chain_id, chain_id, chain_id, chain_id, chain_id]).fetchdf()

    if features_df.empty:
        # Still write empty parquet with correct schema
        features_df = pd.DataFrame()

    # Write output
    output_str = str(output_path)
    if output_str.endswith(".csv"):
        features_df.to_csv(output_path, index=False)
    else:
        features_df.to_parquet(output_path, index=False)

    return output_path
