"""Community detection algorithms replacing community_clustering.ipynb logic.

Uses igraph for Louvain (10-50x faster than NetworkX on large graphs).
"""

from __future__ import annotations

import igraph as ig
import networkx as nx
import pandas as pd


def _nx_to_igraph_undirected(G: nx.DiGraph) -> tuple[ig.Graph, list[str]]:
    """Convert NetworkX DiGraph to undirected igraph for community detection."""
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    ig_g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_g.es["weight"] = weights
    return ig_g, nodes


def louvain_communities(
    G: nx.DiGraph,
    resolution: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Louvain community detection using igraph (much faster than NetworkX).

    Returns DataFrame with address and community_id columns.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "community_id"])

    ig_g, nodes = _nx_to_igraph_undirected(G)
    partition = ig_g.community_multilevel(weights="weight", return_levels=False)

    membership = partition.membership
    df = pd.DataFrame({"address": nodes, "community_id": membership})
    return df


def label_propagation(G: nx.DiGraph) -> pd.DataFrame:
    """Label propagation community detection using igraph."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "community_id"])

    ig_g, nodes = _nx_to_igraph_undirected(G)
    partition = ig_g.community_label_propagation(weights="weight")

    membership = partition.membership
    df = pd.DataFrame({"address": nodes, "community_id": membership})
    return df


def community_stats(
    G: nx.DiGraph,
    community_df: pd.DataFrame,
    max_communities: int = 500,
) -> pd.DataFrame:
    """Compute statistics for each community. Optimized for large numbers of communities.

    Only computes detailed stats for the top `max_communities` by size.
    """
    if community_df.empty:
        return pd.DataFrame()

    # Get community sizes first (cheap)
    sizes = community_df.groupby("community_id").size().reset_index(name="size")
    sizes = sizes.sort_values("size", ascending=False).reset_index(drop=True)

    # Only compute expensive stats for top communities
    top_cids = sizes.head(max_communities)["community_id"].values

    # Build lookup: community_id -> set of members
    comm_members = community_df.groupby("community_id")["address"].apply(set).to_dict()

    stats = []
    for cid in top_cids:
        members = comm_members[cid]
        size = len(members)

        if size <= 1:
            stats.append({
                "community_id": cid, "size": size,
                "internal_edges": 0, "internal_value": 0.0, "density": 0.0,
            })
            continue

        # Use subgraph view (no copy)
        subgraph = G.subgraph(members)
        n_edges = subgraph.number_of_edges()

        internal_value = sum(
            d.get("weight", 0) for _, _, d in subgraph.edges(data=True)
        ) if n_edges > 0 else 0.0

        stats.append({
            "community_id": cid,
            "size": size,
            "internal_edges": n_edges,
            "internal_value": internal_value,
            "density": n_edges / (size * (size - 1)) if size > 1 else 0.0,
        })

    # Append remaining communities with just size info
    if len(sizes) > max_communities:
        remaining = sizes.iloc[max_communities:]
        for _, row in remaining.iterrows():
            stats.append({
                "community_id": row["community_id"],
                "size": row["size"],
                "internal_edges": 0,
                "internal_value": 0.0,
                "density": 0.0,
            })

    df = pd.DataFrame(stats)
    return df.sort_values("size", ascending=False).reset_index(drop=True)
