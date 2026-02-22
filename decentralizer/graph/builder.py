"""Build NetworkX graphs from transaction data. Optimized for large datasets."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import duckdb


def build_graph(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    financial_only: bool = False,
    weighted: bool = True,
) -> nx.DiGraph:
    """Build a directed graph directly from DuckDB with SQL aggregation."""
    where = "WHERE chain_id = ?"
    params = [chain_id]
    if financial_only:
        where += " AND value > 0"

    # Aggregate in SQL — much faster than pandas groupby
    agg = conn.execute(f"""
        SELECT sender, receiver,
               SUM(value) as weight,
               COUNT(*) as tx_count
        FROM transactions
        {where}
        GROUP BY sender, receiver
    """, params).fetchdf()

    return _build_from_agg(agg, weighted)


def build_graph_from_dataframe(
    df: pd.DataFrame,
    weighted: bool = True,
) -> nx.DiGraph:
    """Build from a DataFrame (for cases where data is already in memory)."""
    if df.empty:
        return nx.DiGraph()

    agg = df.groupby(["sender", "receiver"], sort=False).agg(
        weight=("value", "sum"),
        tx_count=("value", "count"),
    ).reset_index()

    return _build_from_agg(agg, weighted)


def _build_from_agg(agg: pd.DataFrame, weighted: bool) -> nx.DiGraph:
    """Build DiGraph from pre-aggregated edge DataFrame."""
    if agg.empty:
        return nx.DiGraph()

    if weighted:
        G = nx.from_pandas_edgelist(
            agg, source="sender", target="receiver",
            edge_attr=["weight", "tx_count"],
            create_using=nx.DiGraph(),
        )
    else:
        G = nx.from_pandas_edgelist(
            agg, source="sender", target="receiver",
            edge_attr=["tx_count"],
            create_using=nx.DiGraph(),
        )
    return G


def graph_stats(G: nx.DiGraph) -> dict:
    """Basic graph statistics."""
    n = G.number_of_nodes()
    return {
        "nodes": n,
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_weakly_connected": nx.is_weakly_connected(G) if n > 0 else False,
        "weakly_connected_components": nx.number_weakly_connected_components(G) if n > 0 else 0,
    }
