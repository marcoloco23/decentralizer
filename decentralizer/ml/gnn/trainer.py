"""Training pipeline for GNN models."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from torch_geometric.data import Data

from decentralizer.ml.gnn.models import (
    AnomalyAutoencoder,
    GATEncoder,
    GNNLinkPredictor,
    GraphSAGEEncoder,
)


def create_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 32,
    **kwargs,
) -> GNNLinkPredictor:
    """Create a GNN link prediction model."""
    if model_type == "graphsage":
        encoder = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_type == "gat":
        encoder = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=kwargs.get("heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'graphsage' or 'gat'.")

    return GNNLinkPredictor(encoder=encoder, embedding_dim=out_channels)


def train_link_prediction(
    model: GNNLinkPredictor,
    split_data: dict,
    epochs: int = 200,
    lr: float = 0.01,
    device: str = "cpu",
) -> dict:
    """Train link prediction model and return metrics."""
    data: Data = split_data["data"].to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state = None
    history = {"train_loss": [], "val_auc": [], "test_metrics": {}}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(data.x, split_data["train_pos"].to(device))

        # Positive edges
        pos_pred = model.predict(z, split_data["train_pos"].to(device))
        # Negative edges
        neg_pred = model.predict(z, split_data["train_neg"].to(device))

        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()
        history["train_loss"].append(loss.item())

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(data.x, split_data["train_pos"].to(device))
                val_auc = _eval_auc(model, z, split_data["val_pos"].to(device), split_data["val_neg"].to(device))
                history["val_auc"].append(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model and evaluate on test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, split_data["train_pos"].to(device))
        test_metrics = _eval_metrics(
            model, z,
            split_data["test_pos"].to(device),
            split_data["test_neg"].to(device),
        )
        history["test_metrics"] = test_metrics

    return history


def _eval_auc(model, z, pos_edge_index, neg_edge_index) -> float:
    pos_pred = torch.sigmoid(model.predict(z, pos_edge_index)).cpu().numpy()
    neg_pred = torch.sigmoid(model.predict(z, neg_edge_index)).cpu().numpy()

    import numpy as np
    labels = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    preds = np.concatenate([pos_pred, neg_pred])
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


def _eval_metrics(model, z, pos_edge_index, neg_edge_index) -> dict:
    pos_pred = torch.sigmoid(model.predict(z, pos_edge_index)).cpu().numpy()
    neg_pred = torch.sigmoid(model.predict(z, neg_edge_index)).cpu().numpy()

    import numpy as np
    labels = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    preds = np.concatenate([pos_pred, neg_pred])
    binary_preds = (preds > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, binary_preds),
        "auc": roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5,
        "mcc": matthews_corrcoef(labels, binary_preds),
    }


def train_anomaly_detector(
    node_embeddings: torch.Tensor,
    embedding_dim: int = 32,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cpu",
) -> tuple[AnomalyAutoencoder, torch.Tensor]:
    """Train anomaly autoencoder and return model + anomaly scores."""
    ae = AnomalyAutoencoder(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)

    z = node_embeddings.to(device)

    for epoch in range(1, epochs + 1):
        ae.train()
        optimizer.zero_grad()
        decoded, _ = ae(z)
        loss = F.mse_loss(decoded, z)
        loss.backward()
        optimizer.step()

    ae.eval()
    with torch.no_grad():
        scores = ae.anomaly_scores(z)

    return ae, scores.cpu()


def save_model(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, weights_only=True))
    return model
