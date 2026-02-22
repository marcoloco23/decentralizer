"""PyG data preparation from NetworkX graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, negative_sampling


def graph_to_pyg_data(G: nx.DiGraph, feature_dim: int = 16) -> Data:
    """Convert a NetworkX DiGraph to PyG Data object.

    Node features are computed from graph structure:
    - in_degree, out_degree, total_degree (normalized)
    - pagerank
    - clustering coefficient
    - weight stats (mean, max incoming/outgoing)
    """
    if G.number_of_nodes() == 0:
        return Data(x=torch.zeros(0, feature_dim), edge_index=torch.zeros(2, 0, dtype=torch.long))

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Compute node features
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    pr = nx.pagerank(G, max_iter=50, tol=1e-4)
    cc = nx.clustering(G)

    # Weight statistics per node
    in_weight = {node: 0.0 for node in nodes}
    out_weight = {node: 0.0 for node in nodes}
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        out_weight[u] += w
        in_weight[v] += w

    features = np.zeros((n, feature_dim), dtype=np.float32)
    max_deg = max(max(in_deg.values(), default=1), max(out_deg.values(), default=1), 1)
    max_weight = max(max(in_weight.values(), default=1), max(out_weight.values(), default=1), 1)

    for i, node in enumerate(nodes):
        features[i, 0] = in_deg.get(node, 0) / max_deg
        features[i, 1] = out_deg.get(node, 0) / max_deg
        features[i, 2] = (in_deg.get(node, 0) + out_deg.get(node, 0)) / (2 * max_deg)
        features[i, 3] = pr.get(node, 0)
        features[i, 4] = cc.get(node, 0)
        features[i, 5] = in_weight.get(node, 0) / max_weight
        features[i, 6] = out_weight.get(node, 0) / max_weight

    # Edge index
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges() if u in node_to_idx and v in node_to_idx]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    # Edge weights
    edge_weights = torch.tensor(
        [G[u][v].get("weight", 1.0) for u, v in G.edges()],
        dtype=torch.float32,
    )

    data = Data(
        x=torch.from_numpy(features),
        edge_index=edge_index,
        edge_attr=edge_weights.unsqueeze(1),
        num_nodes=n,
    )
    data.node_ids = nodes
    data.node_to_idx = node_to_idx

    return data


def prepare_link_prediction_data(
    data: Data,
    test_ratio: float = 0.15,
    val_ratio: float = 0.05,
    neg_ratio: float = 1.0,
) -> dict:
    """Split edges into train/val/test and generate negative samples."""
    num_edges = data.edge_index.size(1)
    perm = torch.randperm(num_edges)

    num_test = int(num_edges * test_ratio)
    num_val = int(num_edges * val_ratio)

    test_idx = perm[:num_test]
    val_idx = perm[num_test:num_test + num_val]
    train_idx = perm[num_test + num_val:]

    train_edge_index = data.edge_index[:, train_idx]
    val_edge_index = data.edge_index[:, val_idx]
    test_edge_index = data.edge_index[:, test_idx]

    # Negative sampling
    num_neg_train = int(len(train_idx) * neg_ratio)
    num_neg_val = int(len(val_idx) * neg_ratio)
    num_neg_test = int(len(test_idx) * neg_ratio)

    neg_train = negative_sampling(
        train_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=num_neg_train,
    )
    neg_val = negative_sampling(
        data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=num_neg_val,
    )
    neg_test = negative_sampling(
        data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=num_neg_test,
    )

    return {
        "data": data,
        "train_pos": train_edge_index,
        "train_neg": neg_train,
        "val_pos": val_edge_index,
        "val_neg": neg_val,
        "test_pos": test_edge_index,
        "test_neg": neg_test,
    }
