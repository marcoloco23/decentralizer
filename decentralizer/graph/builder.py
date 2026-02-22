"""Build NetworkX graphs from transaction data."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import duckdb

from decentralizer.storage.database import get_edge_dataframe


def build_graph(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    financial_only: bool = False,
    weighted: bool = True,
) -> nx.DiGraph:
    """Build a directed graph from transaction edges.

    Multiple transactions between same pair are aggregated:
    - weight = total value transferred
    - tx_count = number of transactions
    """
    df = get_edge_dataframe(conn, chain_id=chain_id, financial_only=financial_only)
    return build_graph_from_dataframe(df, weighted=weighted)


def build_graph_from_dataframe(
    df: pd.DataFrame,
    weighted: bool = True,
) -> nx.DiGraph:
    """Build a directed graph from an edge DataFrame with sender/receiver columns."""
    if df.empty:
        return nx.DiGraph()

    # Aggregate multiple edges between same pair
    agg = df.groupby(["sender", "receiver"]).agg(
        total_value=("value", "sum"),
        tx_count=("value", "count"),
        avg_gas=("gas", "mean"),
    ).reset_index()

    G = nx.DiGraph()
    for _, row in agg.iterrows():
        attrs = {"tx_count": int(row["tx_count"]), "avg_gas": row["avg_gas"]}
        if weighted:
            attrs["weight"] = row["total_value"]
        G.add_edge(row["sender"], row["receiver"], **attrs)

    return G


def build_multigraph(
    conn: duckdb.DuckDBPyConnection,
    chain_id: int = 1,
    financial_only: bool = False,
) -> nx.MultiDiGraph:
    """Build a multi-edge directed graph (one edge per transaction)."""
    df = get_edge_dataframe(conn, chain_id=chain_id, financial_only=financial_only)
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        G.add_edge(
            row["sender"],
            row["receiver"],
            value=row["value"],
            gas=row["gas"],
            gas_price=row["gas_price"],
            block_number=row["block_number"],
            timestamp=row["timestamp"],
        )
    return G


def graph_stats(G: nx.DiGraph) -> dict:
    """Basic graph statistics."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_weakly_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
        "weakly_connected_components": nx.number_weakly_connected_components(G),
    }
