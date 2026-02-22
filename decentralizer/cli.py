"""Click CLI: fetch, migrate, train, dashboard, analyze."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from decentralizer.config import PROJECT_ROOT


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """Decentralizer - Multi-chain blockchain intelligence platform."""
    pass


@cli.command()
@click.option("--chain", default="ethereum", help="Chain name or ID (ethereum, arbitrum, optimism, base, polygon)")
@click.option("--blocks", default=100, help="Number of blocks to fetch")
@click.option("--concurrency", default=10, help="Max concurrent requests")
def fetch(chain: str, blocks: int, concurrency: int):
    """Fetch blockchain data from an EVM chain."""
    from decentralizer.chain.registry import resolve_chain
    from decentralizer.chain.fetcher import BlockFetcher
    from decentralizer.storage.database import get_connection, insert_blocks, insert_transactions

    chain_config = resolve_chain(chain)
    click.echo(f"Fetching {blocks} blocks from {chain_config.name} (chain_id={chain_config.chain_id})...")

    fetcher = BlockFetcher(chain_config.chain_id, max_concurrent=concurrency)

    async def _fetch():
        return await fetcher.fetch_latest(num_blocks=blocks)

    block_models, tx_models = asyncio.run(_fetch())

    conn = get_connection()
    blocks_df = fetcher.blocks_to_dataframe(block_models)
    txs_df = fetcher.transactions_to_dataframe(tx_models)

    n_blocks = insert_blocks(conn, blocks_df)
    n_txs = insert_transactions(conn, txs_df)

    click.echo(f"Inserted {n_blocks} blocks and {n_txs} transactions into DuckDB.")
    conn.close()


@cli.command()
def migrate():
    """Import legacy CSV data into DuckDB."""
    from decentralizer.storage.database import get_connection, migrate_legacy_csvs

    click.echo("Migrating legacy CSV data to DuckDB...")
    conn = get_connection()
    counts = migrate_legacy_csvs(conn)

    for label, count in counts.items():
        click.echo(f"  {label}: {count} transactions imported")

    total = sum(counts.values())
    click.echo(f"Total: {total} transactions migrated.")
    conn.close()


@cli.command()
@click.option("--model", "model_type", default="graphsage", type=click.Choice(["graphsage", "gat"]))
@click.option("--epochs", default=200)
@click.option("--lr", default=0.01)
@click.option("--chain-id", default=1)
def train(model_type: str, epochs: int, lr: float, chain_id: int):
    """Train GNN models for link prediction and anomaly detection."""
    import torch
    import pandas as pd
    from decentralizer.storage.database import get_connection
    from decentralizer.graph.builder import build_graph
    from decentralizer.ml.gnn.embeddings import graph_to_pyg_data, prepare_link_prediction_data
    from decentralizer.ml.gnn.trainer import (
        create_model,
        train_link_prediction,
        train_anomaly_detector,
        save_model,
    )

    conn = get_connection()
    click.echo(f"Building graph for chain_id={chain_id}...")
    G = build_graph(conn, chain_id=chain_id, financial_only=True)
    click.echo(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_nodes() < 10:
        click.echo("Graph too small to train. Fetch more data first.")
        return

    click.echo("Preparing PyG data...")
    data = graph_to_pyg_data(G)
    split = prepare_link_prediction_data(data)

    click.echo(f"Creating {model_type} model...")
    model = create_model(model_type, in_channels=data.x.size(1))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    click.echo(f"Training on {device} for {epochs} epochs...")

    history = train_link_prediction(model, split, epochs=epochs, lr=lr, device=device)

    metrics = history["test_metrics"]
    click.echo(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, MCC: {metrics['mcc']:.4f}")

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, model_dir / f"{model_type}_link_predictor.pt")
    click.echo(f"Model saved to {model_dir / f'{model_type}_link_predictor.pt'}")

    # Anomaly detection
    click.echo("Training anomaly detector...")
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x.to(device), split["train_pos"].to(device))
    ae, scores = train_anomaly_detector(z, embedding_dim=z.size(1), device=device)

    anom_df = pd.DataFrame({"address": data.node_ids, "anomaly_score": scores.numpy()})
    anom_df = anom_df.sort_values("anomaly_score", ascending=False)
    anom_df.to_csv(model_dir / "anomaly_scores.csv", index=False)
    save_model(ae, model_dir / "anomaly_autoencoder.pt")
    click.echo(f"Anomaly scores saved for {len(anom_df)} addresses.")
    conn.close()


@cli.command()
@click.option("--port", default=8501)
def dashboard(port: int):
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=True,
    )


@cli.command()
@click.option("--chain-id", default=1)
@click.option("--top-k", default=50)
@click.option("--financial-only", is_flag=True, default=True)
def analyze(chain_id: int, top_k: int, financial_only: bool):
    """Run all graph algorithms and output results."""
    from decentralizer.storage.database import get_connection, upsert_address_metrics
    from decentralizer.graph.builder import build_graph, graph_stats
    from decentralizer.graph import algorithms as algo
    from decentralizer.graph.communities import louvain_communities, community_stats
    import pandas as pd

    conn = get_connection()
    click.echo(f"Building graph for chain_id={chain_id}...")
    G = build_graph(conn, chain_id=chain_id, financial_only=financial_only)
    stats = graph_stats(G)
    click.echo(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges, {stats['weakly_connected_components']} components")

    if G.number_of_nodes() == 0:
        click.echo("No data in graph. Run `decentralizer migrate` or `decentralizer fetch` first.")
        return

    click.echo("\n--- PageRank ---")
    pr_df = algo.page_rank(G, top_k=top_k)
    click.echo(pr_df.head(10).to_string())

    click.echo("\n--- Weighted PageRank ---")
    wpr_df = algo.weighted_page_rank(G, top_k=top_k)
    click.echo(wpr_df.head(10).to_string())

    click.echo("\n--- Influence Maximization ---")
    inf_df = algo.max_influence(G, top_k=min(20, top_k))
    click.echo(inf_df.to_string())

    click.echo("\n--- Betweenness Centrality ---")
    k_sample = min(500, G.number_of_nodes()) if G.number_of_nodes() > 5000 else None
    bc_df = algo.betweenness_centrality(G, top_k=top_k, k_sample=k_sample)
    click.echo(bc_df.head(10).to_string())

    click.echo("\n--- Community Detection (Louvain) ---")
    comm_df = louvain_communities(G)
    cstats = community_stats(G, comm_df)
    click.echo(f"Found {len(cstats)} communities")
    click.echo(cstats.head(10).to_string())

    # Save metrics to DB
    click.echo("\nSaving metrics to database...")
    metrics_df = pr_df[["address", "page_rank"]].copy()
    metrics_df = metrics_df.merge(wpr_df[["address", "weighted_page_rank"]], on="address", how="outer")
    metrics_df = metrics_df.merge(bc_df[["address", "betweenness_centrality"]], on="address", how="outer")
    metrics_df = metrics_df.merge(comm_df[["address", "community_id"]], on="address", how="outer")
    metrics_df["chain_id"] = chain_id
    metrics_df = metrics_df.fillna(0)

    # Ensure column order matches table
    metrics_df["clustering_coefficient"] = 0.0
    metrics_df["influence_score"] = 0.0
    metrics_df["anomaly_score"] = 0.0

    cols = ["chain_id", "address", "page_rank", "weighted_page_rank",
            "betweenness_centrality", "clustering_coefficient",
            "influence_score", "community_id", "anomaly_score"]
    metrics_df = metrics_df[cols]

    upsert_address_metrics(conn, metrics_df)
    click.echo(f"Saved metrics for {len(metrics_df)} addresses.")
    conn.close()


if __name__ == "__main__":
    cli()
