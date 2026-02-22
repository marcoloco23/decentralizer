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

import igraph as ig
import networkx as nx
import pandas as pd


def _nx_to_igraph(G: nx.DiGraph) -> tuple[ig.Graph, list[str]]:
    """Convert NetworkX DiGraph to igraph for fast algorithms."""
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    ig_g = ig.Graph(n=len(nodes), edges=edges, directed=True)
    ig_g.es["weight"] = weights
    return ig_g, nodes


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
    df = pd.DataFrame(list(scores.items()), columns=["address", "page_rank"])
    df = df.nlargest(top_k, "page_rank").reset_index(drop=True)
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
    df = pd.DataFrame(list(scores.items()), columns=["address", "weighted_page_rank"])
    df = df.nlargest(top_k, "weighted_page_rank").reset_index(drop=True)
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

    valid_sources = [a for a in source_addresses if a in G]
    if not valid_sources:
        return pd.DataFrame(columns=["address", "score"])

    weight = 1.0 / len(valid_sources)
    personalization = {addr: weight for addr in valid_sources}

    scores = nx.pagerank(
        G, alpha=damping, personalization=personalization,
        max_iter=max_iter, tol=tol,
    )
    df = pd.DataFrame(list(scores.items()), columns=["address", "score"])
    return df.nlargest(top_k, "score").reset_index(drop=True)


def max_influence(
    G: nx.DiGraph,
    top_k: int = 100,
) -> pd.DataFrame:
    """Greedy influence maximization (replaces maxInfluence.gsql).

    Optimized: pre-sort by out-degree, use set for selected lookup.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "influence_score"])

    # Pre-sort candidates by out-degree (highest first) for faster greedy selection
    candidates = sorted(G.nodes(), key=lambda n: G.out_degree(n), reverse=True)

    influenced: set[str] = set()
    selected: set[str] = set()
    results: list[dict] = []

    for _ in range(top_k):
        best_node = None
        best_reach = 0

        for node in candidates:
            if node in selected:
                continue
            # Count successors not yet influenced
            reach = sum(1 for n in G.successors(node) if n not in influenced)
            if reach > best_reach:
                best_reach = reach
                best_node = node

            # Early exit: if a node has out-degree <= best_reach, skip rest
            # (since candidates are sorted by out-degree descending)
            if G.out_degree(node) <= best_reach and best_node is not None:
                break

        if best_node is None or best_reach == 0:
            break

        results.append({"address": best_node, "influence_score": float(best_reach)})
        selected.add(best_node)
        influenced.add(best_node)
        influenced.update(G.successors(best_node))

    return pd.DataFrame(results)


def recommend_addresses(
    G: nx.DiGraph,
    source_address: str,
    top_k: int = 100,
) -> pd.DataFrame:
    """Address recommendation via cosine similarity (replaces recommendAddresses.gsql).

    Optimized: cache successor sets, avoid recomputation.
    """
    if source_address not in G:
        return pd.DataFrame(columns=["address", "similarity_score"])

    source_neighbors = set(G.successors(source_address))
    source_degree = len(source_neighbors)
    if source_degree == 0:
        return pd.DataFrame(columns=["address", "similarity_score"])

    # Cache successor sets for candidates
    successor_cache: dict[str, set] = {}
    candidates: dict[str, float] = {}

    for neighbor in source_neighbors:
        for candidate in G.successors(neighbor):
            if candidate == source_address or candidate in source_neighbors:
                continue
            if candidate not in successor_cache:
                successor_cache[candidate] = set(G.successors(candidate))
            candidate_neighbors = successor_cache[candidate]
            common = len(source_neighbors & candidate_neighbors)
            if common > 0:
                candidate_degree = len(candidate_neighbors)
                score = common / math.sqrt(source_degree * candidate_degree)
                candidates[candidate] = candidates.get(candidate, 0) + score

    if not candidates:
        return pd.DataFrame(columns=["address", "similarity_score"])

    df = pd.DataFrame(list(candidates.items()), columns=["address", "similarity_score"])
    return df.nlargest(top_k, "similarity_score").reset_index(drop=True)


def data_overview(G: nx.DiGraph) -> pd.DataFrame:
    """Edge list overview (replaces dataOverview.gsql)."""
    if G.number_of_edges() == 0:
        return pd.DataFrame()
    return nx.to_pandas_edgelist(G)


def address_degree(G: nx.DiGraph, address: str) -> dict:
    """Degree info for an address (replaces test.gsql)."""
    if address not in G:
        return {"address": address, "in_degree": 0, "out_degree": 0, "total_degree": 0}
    ind = G.in_degree(address)
    outd = G.out_degree(address)
    return {
        "address": address,
        "in_degree": ind,
        "out_degree": outd,
        "total_degree": ind + outd,
    }


# --- Additional algorithms ---

def betweenness_centrality(
    G: nx.DiGraph,
    top_k: int = 100,
    k_sample: int | None = None,
) -> pd.DataFrame:
    """Betweenness centrality using igraph.

    For large graphs (>10k nodes), automatically uses approximate estimation
    by sampling source vertices, unless exact=True via k_sample=None with small graph.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "betweenness_centrality"])

    ig_g, nodes = _nx_to_igraph(G)
    n = len(nodes)

    # For large graphs, use cutoff-based approximation
    if k_sample is not None or n > 10000:
        cutoff = 3  # Only consider paths up to length 3
        bc = ig_g.betweenness(directed=True, cutoff=cutoff)
        norm = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    else:
        bc = ig_g.betweenness(directed=True)
        norm = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0

    df = pd.DataFrame({"address": nodes, "betweenness_centrality": [b * norm for b in bc]})
    return df.nlargest(top_k, "betweenness_centrality").reset_index(drop=True)


def clustering_coefficients(
    G: nx.DiGraph,
    top_k: int = 100,
) -> pd.DataFrame:
    """Clustering coefficients using igraph (much faster than NetworkX)."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "clustering_coefficient"])

    ig_g, nodes = _nx_to_igraph(G)
    cc = ig_g.transitivity_local_undirected(mode="zero")
    df = pd.DataFrame({"address": nodes, "clustering_coefficient": cc})
    return df.nlargest(top_k, "clustering_coefficient").reset_index(drop=True)


def k_core_decomposition(G: nx.DiGraph) -> pd.DataFrame:
    """K-core decomposition on undirected projection."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["address", "core_number"])

    U = G.to_undirected()
    cores = nx.core_number(U)
    df = pd.DataFrame(list(cores.items()), columns=["address", "core_number"])
    return df.sort_values("core_number", ascending=False).reset_index(drop=True)
