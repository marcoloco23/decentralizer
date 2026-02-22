"""GNN models: GraphSAGE, GAT, LinkPredictor, AnomalyDetector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATEncoder(nn.Module):
    """GAT v2 encoder for node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32, heads: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class LinkPredictor(nn.Module):
    """MLP link predictor on top of node embeddings."""

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Predict link probability between source and destination embeddings."""
        h = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(h).squeeze(-1)


class GNNLinkPredictor(nn.Module):
    """End-to-end GNN + link predictor model."""

    def __init__(self, encoder: nn.Module, embedding_dim: int = 32):
        super().__init__()
        self.encoder = encoder
        self.predictor = LinkPredictor(embedding_dim=embedding_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def predict(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return self.predictor(z[src], z[dst])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pred_edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.predict(z, pred_edge_index)


class AnomalyAutoencoder(nn.Module):
    """Autoencoder for anomaly detection on node embeddings.

    High reconstruction error indicates anomalous behavior.
    """

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder_net(z)
        decoded = self.decoder_net(encoded)
        return decoded, encoded

    def anomaly_scores(self, z: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as reconstruction error per node."""
        decoded, _ = self.forward(z)
        return torch.mean((z - decoded) ** 2, dim=-1)
