"""Graph algorithms replacing all 7 TigerGraph GSQL queries.

Mapping:
  pageRank.gsql              -> page_rank()
  weightedPageRank.gsql      -> weighted_page_rank()
  personalizedPageRank.gsql  -> personalized_page_rank()
  maxInfluence.gsql          -> max_influence()
  recommendAddresses.gsql    -> recommend_addresses()
  dataOverview.gsql          -> data_overview()
  test.gsql                  -> address_degree()

Additional algorithms:
  - betweenness_centrality()
  - clustering_coefficients()
  - k_core_decomposition()
"""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd


def page_rank(
    G: nx.DiGraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    top_k: int = 100,
) -> pd.DataFrame:
    """Standard PageRank (replaces pageRank.gsql)."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "page_rank"])

    scores = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=tol)
    df = pd.DataFrame(
        [{"address": addr, "page_rank": score} for addr, score in scores.items()]
    )
    df = df.sort_values("page_rank", ascending=False).head(top_k).reset_index(drop=True)
    # Normalize to percentage
    total = df["page_rank"].sum()
    if total > 0:
        df["page_rank_pct"] = df["page_rank"] / total * 100
    return df


def weighted_page_rank(
    G: nx.DiGraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    top_k: int = 100,
) -> pd.DataFrame:
    """PageRank weighted by transaction value (replaces weightedPageRank.gsql)."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "weighted_page_rank"])

    scores = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=tol, weight="weight")
    df = pd.DataFrame(
        [{"address": addr, "weighted_page_rank": score} for addr, score in scores.items()]
    )
    df = df.sort_values("weighted_page_rank", ascending=False).head(top_k).reset_index(drop=True)
    total = df["weighted_page_rank"].sum()
    if total > 0:
        df["weighted_page_rank_pct"] = df["weighted_page_rank"] / total * 100
    return df


def personalized_page_rank(
    G: nx.DiGraph,
    source_addresses: list[str],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    top_k: int = 100,
) -> pd.DataFrame:
    """Personalized PageRank from seed addresses (replaces personalizedPageRank.gsql)."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "score"])

    # Build personalization dict: uniform weight on source addresses
    personalization = {}
    valid_sources = [a for a in source_addresses if a in G]
    if not valid_sources:
        return pd.DataFrame(columns=["address", "score"])

    weight = 1.0 / len(valid_sources)
    for addr in valid_sources:
        personalization[addr] = weight

    scores = nx.pagerank(
        G, alpha=damping, personalization=personalization,
        max_iter=max_iter, tol=tol,
    )
    df = pd.DataFrame(
        [{"address": addr, "score": score} for addr, score in scores.items()]
    )
    return df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


def max_influence(
    G: nx.DiGraph,
    top_k: int = 100,
) -> pd.DataFrame:
    """Greedy influence maximization (replaces maxInfluence.gsql).

    Iteratively selects the node that reaches the most uninfluenced nodes.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "influence_score"])

    influenced: set[str] = set()
    selected: list[dict] = []

    for _ in range(top_k):
        best_node = None
        best_reach = 0

        for node in G.nodes():
            if node in {s["address"] for s in selected}:
                continue
            # Count successors not yet influenced
            reach = sum(1 for n in G.successors(node) if n not in influenced)
            if reach > best_reach:
                best_reach = reach
                best_node = node

        if best_node is None or best_reach == 0:
            break

        selected.append({"address": best_node, "influence_score": float(best_reach)})
        influenced.add(best_node)
        influenced.update(G.successors(best_node))

    return pd.DataFrame(selected)


def recommend_addresses(
    G: nx.DiGraph,
    source_address: str,
    top_k: int = 100,
) -> pd.DataFrame:
    """Address recommendation via cosine similarity (replaces recommendAddresses.gsql).

    Finds addresses with similar transaction patterns using neighbor overlap.
    Score = common_neighbors / sqrt(source_degree * candidate_degree)
    """
    if source_address not in G:
        return pd.DataFrame(columns=["address", "similarity_score"])

    source_neighbors = set(G.successors(source_address))
    source_degree = len(source_neighbors)
    if source_degree == 0:
        return pd.DataFrame(columns=["address", "similarity_score"])

    # Find 2-hop neighbors (friends of friends)
    candidates: dict[str, float] = {}
    for neighbor in source_neighbors:
        for candidate in G.successors(neighbor):
            if candidate == source_address or candidate in source_neighbors:
                continue
            candidate_neighbors = set(G.successors(candidate))
            common = len(source_neighbors & candidate_neighbors)
            if common > 0:
                candidate_degree = len(candidate_neighbors)
                score = common / math.sqrt(source_degree * candidate_degree)
                # Accumulate scores from multiple paths
                candidates[candidate] = candidates.get(candidate, 0) + score

    df = pd.DataFrame(
        [{"address": addr, "similarity_score": score}
         for addr, score in candidates.items()]
    )
    return df.sort_values("similarity_score", ascending=False).head(top_k).reset_index(drop=True)


def data_overview(G: nx.DiGraph) -> pd.DataFrame:
    """Edge list overview (replaces dataOverview.gsql)."""
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({"from": u, "to": v, **data})
    return pd.DataFrame(edges)


def address_degree(G: nx.DiGraph, address: str) -> dict:
    """Degree info for an address (replaces test.gsql)."""
    if address not in G:
        return {"address": address, "in_degree": 0, "out_degree": 0, "total_degree": 0}
    return {
        "address": address,
        "in_degree": G.in_degree(address),
        "out_degree": G.out_degree(address),
        "total_degree": G.in_degree(address) + G.out_degree(address),
    }


# --- Additional algorithms ---

def betweenness_centrality(
    G: nx.DiGraph,
    top_k: int = 100,
    k_sample: int | None = None,
) -> pd.DataFrame:
    """Betweenness centrality. Use k_sample for approximate results on large graphs."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "betweenness_centrality"])

    bc = nx.betweenness_centrality(G, k=k_sample)
    df = pd.DataFrame(
        [{"address": addr, "betweenness_centrality": score}
         for addr, score in bc.items()]
    )
    return df.sort_values("betweenness_centrality", ascending=False).head(top_k).reset_index(drop=True)


def clustering_coefficients(
    G: nx.DiGraph,
    top_k: int = 100,
) -> pd.DataFrame:
    """Clustering coefficients for directed graph."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "clustering_coefficient"])

    cc = nx.clustering(G)
    df = pd.DataFrame(
        [{"address": addr, "clustering_coefficient": score}
         for addr, score in cc.items()]
    )
    return df.sort_values("clustering_coefficient", ascending=False).head(top_k).reset_index(drop=True)


def k_core_decomposition(G: nx.DiGraph) -> pd.DataFrame:
    """K-core decomposition on undirected projection."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "core_number"])

    U = G.to_undirected()
    cores = nx.core_number(U)
    df = pd.DataFrame(
        [{"address": addr, "core_number": k} for addr, k in cores.items()]
    )
    return df.sort_values("core_number", ascending=False).reset_index(drop=True)
