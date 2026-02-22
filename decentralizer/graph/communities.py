"""Community detection algorithms replacing community_clustering.ipynb logic."""

from __future__ import annotations

import networkx as nx
import pandas as pd


def louvain_communities(
    G: nx.DiGraph,
    resolution: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Louvain community detection on undirected projection.

    Returns DataFrame with address and community_id columns.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "community_id"])

    U = G.to_undirected()
    communities = nx.community.louvain_communities(U, resolution=resolution, seed=seed)

    rows = []
    for community_id, members in enumerate(communities):
        for address in members:
            rows.append({"address": address, "community_id": community_id})

    return pd.DataFrame(rows)


def label_propagation(G: nx.DiGraph) -> pd.DataFrame:
    """Label propagation community detection on undirected projection."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "community_id"])

    U = G.to_undirected()
    communities = nx.community.label_propagation_communities(U)

    rows = []
    for community_id, members in enumerate(communities):
        for address in members:
            rows.append({"address": address, "community_id": community_id})

    return pd.DataFrame(rows)


def community_stats(
    G: nx.DiGraph,
    community_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute statistics for each community."""
    if community_df.empty:
        return pd.DataFrame()

    stats = []
    for cid, group in community_df.groupby("community_id"):
        members = set(group["address"])
        subgraph = G.subgraph(members)

        # Calculate total value flowing within community
        internal_value = sum(
            data.get("weight", 0) for _, _, data in subgraph.edges(data=True)
        )

        stats.append({
            "community_id": cid,
            "size": len(members),
            "internal_edges": subgraph.number_of_edges(),
            "internal_value": internal_value,
            "density": nx.density(subgraph) if len(members) > 1 else 0,
        })

    df = pd.DataFrame(stats)
    return df.sort_values("size", ascending=False).reset_index(drop=True)
