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
@click.option("--concurrency", default=50, help="Max concurrent requests")
def fetch(chain: str, blocks: int, concurrency: int):
    """Fetch blockchain data from an EVM chain."""
    from decentralizer.chain.registry import resolve_chain
    from decentralizer.chain.fetcher import BlockFetcher
    from decentralizer.storage.database import get_connection, insert_blocks, insert_transactions

    chain_config = resolve_chain(chain)
    click.echo(f"Fetching {blocks} blocks from {chain_config.name} (chain_id={chain_config.chain_id}, concurrency={concurrency})...")

    fetcher = BlockFetcher(chain_config.chain_id, max_concurrent=concurrency)

    async def _fetch():
        return await fetcher.fetch_latest(num_blocks=blocks)

    blocks_df, txs_df = asyncio.run(_fetch())

    conn = get_connection()

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


@cli.command("fetch-transfers")
@click.option("--chain", default="ethereum", help="Chain name or ID")
@click.option("--blocks", default=100, help="Number of latest blocks to fetch transfers from")
@click.option("--start-block", default=None, type=int, help="Start block (overrides --blocks)")
@click.option("--end-block", default=None, type=int, help="End block (requires --start-block)")
@click.option("--batch-size", default=500, help="Blocks per eth_getLogs call")
@click.option("--concurrency", default=50, help="Max concurrent requests")
def fetch_transfers(chain: str, blocks: int, start_block: int | None, end_block: int | None, batch_size: int, concurrency: int):
    """Fetch ERC-20 token transfer events from the chain."""
    from decentralizer.chain.registry import resolve_chain
    from decentralizer.chain.fetcher import BlockFetcher
    from decentralizer.storage.database import get_connection, insert_token_transfers

    chain_config = resolve_chain(chain)
    fetcher = BlockFetcher(chain_config.chain_id, max_concurrent=concurrency)

    async def _fetch():
        if start_block is not None and end_block is not None:
            return start_block, end_block, await fetcher.fetch_token_transfers(start_block, end_block, batch_size=batch_size)
        else:
            latest = await fetcher.provider.get_latest_block_number()
            s = latest - blocks + 1
            return s, latest, await fetcher.fetch_token_transfers(s, latest, batch_size=batch_size)

    s, e, transfers_df = asyncio.run(_fetch())
    click.echo(f"Fetched {len(transfers_df)} transfer events from blocks {s}-{e}")

    if not transfers_df.empty:
        conn = get_connection()
        n = insert_token_transfers(conn, transfers_df)
        click.echo(f"Inserted {n} token transfers into DuckDB.")

        # Backfill timestamps from blocks table
        from decentralizer.tokens.transfers import backfill_timestamps
        backfill_timestamps(conn, chain_config.chain_id)

        conn.close()


@cli.command("resolve-tokens")
@click.option("--chain-id", default=1)
def resolve_tokens(chain_id: int):
    """Resolve token metadata (symbol, decimals) for all tokens in token_transfers."""
    from decentralizer.storage.database import get_connection, upsert_token_metadata
    from decentralizer.labels.etherscan import resolve_token_metadata
    from decentralizer.chain.provider import ChainProvider
    from decentralizer.tokens.transfers import backfill_decimals

    conn = get_connection()

    # Get unique token addresses not yet in metadata
    tokens = conn.execute("""
        SELECT DISTINCT tt.token_address
        FROM token_transfers tt
        LEFT JOIN token_metadata tm
          ON tt.chain_id = tm.chain_id AND tt.token_address = tm.address
        WHERE tt.chain_id = ? AND tm.address IS NULL
    """, [chain_id]).fetchdf()

    if tokens.empty:
        click.echo("All tokens already have metadata.")
        conn.close()
        return

    addresses = tokens["token_address"].tolist()
    click.echo(f"Resolving metadata for {len(addresses)} tokens...")

    provider = ChainProvider(chain_id)

    async def _resolve():
        return await resolve_token_metadata(chain_id, addresses, w3=provider.w3)

    meta_df = asyncio.run(_resolve())
    if not meta_df.empty:
        upsert_token_metadata(conn, meta_df)
        click.echo(f"Saved metadata for {len(meta_df)} tokens.")

    # Backfill decimal values
    backfill_decimals(conn, chain_id)
    click.echo("Backfilled value_decimal in token_transfers.")
    conn.close()


@cli.command("backfill-prices")
@click.option("--chain-id", default=1)
@click.option("--rate-limit", default=1.0, help="Seconds between DeFiLlama API calls")
def backfill_prices_cmd(chain_id: int, rate_limit: float):
    """Fetch historical token prices from DeFiLlama."""
    from decentralizer.storage.database import get_connection
    from decentralizer.tokens.pricing import backfill_prices

    conn = get_connection()
    click.echo(f"Backfilling prices for chain_id={chain_id}...")

    async def _backfill():
        return await backfill_prices(conn, chain_id=chain_id, rate_limit=rate_limit)

    n = asyncio.run(_backfill())
    click.echo(f"Inserted {n} price records.")
    conn.close()


@cli.command()
@click.option("--chain-id", default=1)
@click.option("--top-k", default=1000)
@click.option("--min-transfers", default=5, help="Min transfers for P&L calculation")
def score(chain_id: int, top_k: int, min_transfers: int):
    """Compute wallet P&L and smart money scores."""
    from decentralizer.storage.database import get_connection, upsert_smart_money_scores
    from decentralizer.scoring.pnl import calculate_all_wallet_pnl
    from decentralizer.scoring.smart_money import compute_smart_money_scores
    from decentralizer.tokens.dex import identify_dex_trades
    from decentralizer.storage.database import insert_dex_trades

    conn = get_connection()

    # Step 1: Reconstruct DEX trades
    click.echo("Reconstructing DEX trades...")
    trades_df = identify_dex_trades(conn, chain_id)
    if not trades_df.empty:
        n = insert_dex_trades(conn, trades_df)
        click.echo(f"Identified {n} DEX trades.")
    else:
        click.echo("No DEX trades found.")

    # Step 2: Calculate P&L for all qualifying wallets
    click.echo(f"Computing wallet P&L (min_transfers={min_transfers})...")
    n_pnl = calculate_all_wallet_pnl(conn, chain_id, min_transfers=min_transfers)
    click.echo(f"Computed P&L for {n_pnl} wallet-token pairs.")

    # Step 3: Compute smart money scores
    click.echo(f"Computing smart money scores (top_k={top_k})...")
    scores_df = compute_smart_money_scores(conn, chain_id, top_k=top_k)
    if not scores_df.empty:
        upsert_smart_money_scores(conn, scores_df)
        click.echo(f"Saved smart money scores for {len(scores_df)} wallets.")
        click.echo("\nTop 10 Smart Money Wallets:")
        click.echo(scores_df.head(10)[["address", "composite_score", "pnl_score", "page_rank_score", "rank"]].to_string())
    else:
        click.echo("No smart money scores computed. Run `analyze` and `fetch-transfers` first.")

    conn.close()


@cli.command()
@click.option("--chain-id", default=1)
@click.option("--output", default="data/features.parquet", help="Output file path (.parquet or .csv)")
def export(chain_id: int, output: str):
    """Export feature dataset for Numerai pipeline."""
    from decentralizer.storage.database import get_connection
    from decentralizer.scoring.export import export_feature_dataset

    conn = get_connection()
    click.echo(f"Exporting feature dataset for chain_id={chain_id}...")

    path = export_feature_dataset(conn, chain_id, output_path=output)
    import pandas as pd
    if output.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    click.echo(f"Exported {len(df)} addresses with {len(df.columns)} features to {path}")
    click.echo(f"Columns: {', '.join(df.columns.tolist())}")
    conn.close()


@cli.command("map-tokens")
@click.option("--live-data", required=True, type=click.Path(exists=True), help="Path to Numerai crypto_live.parquet")
@click.option("--rate-limit", default=3.0, help="Seconds between CoinGecko API calls")
def map_tokens(live_data: str, rate_limit: float):
    """Map Numerai symbols to on-chain contract addresses via CoinGecko."""
    from decentralizer.storage.database import get_connection
    from decentralizer.features.token_mapping import build_token_mapping

    conn = get_connection()
    click.echo(f"Building token mapping from {live_data}...")

    async def _map():
        return await build_token_mapping(conn, live_data, rate_limit=rate_limit)

    mapping_df = asyncio.run(_map())

    if mapping_df.empty:
        click.echo("No tokens mapped. Check CoinGecko connectivity.")
    else:
        n_symbols = mapping_df["symbol"].nunique()
        n_chains = mapping_df["chain_id"].nunique()
        click.echo(f"Mapped {n_symbols} symbols across {n_chains} chains ({len(mapping_df)} total entries).")

        # Show Ethereum mappings
        eth_count = len(mapping_df[mapping_df["chain_id"] == 1])
        click.echo(f"  Ethereum (chain_id=1): {eth_count} tokens")

    conn.close()


@cli.command("compute-features")
@click.option("--chain-id", default=1)
@click.option("--date", "as_of_date", default=None, help="Date to compute features for (YYYY-MM-DD, default: today)")
def compute_features(chain_id: int, as_of_date: str | None):
    """Compute on-chain token features for all mapped tokens."""
    from datetime import date as dt_date
    from decentralizer.storage.database import get_connection
    from decentralizer.features.on_chain import compute_all_token_features

    conn = get_connection()

    if as_of_date:
        target_date = dt_date.fromisoformat(as_of_date)
    else:
        target_date = dt_date.today()

    click.echo(f"Computing on-chain features for chain_id={chain_id} as of {target_date}...")

    features_df = compute_all_token_features(conn, chain_id, as_of_date=target_date)

    if features_df.empty:
        click.echo("No features computed. Run `map-tokens` and `fetch-transfers` first.")
    else:
        non_zero = (features_df.select_dtypes(include="number").fillna(0) != 0).any(axis=1).sum()
        click.echo(f"Computed features for {len(features_df)} tokens ({non_zero} with non-zero data).")

    conn.close()


@cli.command("export-features")
@click.option("--chain-id", default=1)
@click.option("--output", default="data/inference_features.parquet", help="Output Parquet path")
@click.option("--upload-s3", is_flag=True, default=False, help="Upload to S3 inference store")
def export_features(chain_id: int, output: str, upload_s3: bool):
    """Export quantile-normalized on-chain features for Numerai inference."""
    from decentralizer.storage.database import get_connection
    from decentralizer.features.export_numerai import export_inference_features, upload_to_s3

    conn = get_connection()
    click.echo(f"Exporting inference features for chain_id={chain_id}...")

    path = export_inference_features(conn, chain_id, output_path=output)

    import pandas as pd
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    click.echo(f"Exported {len(df)} symbols with {len(feature_cols)} features to {path}")

    if feature_cols:
        click.echo(f"Feature columns: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

    if upload_s3:
        click.echo("Uploading to S3...")
        s3_uri = upload_to_s3(path)
        click.echo(f"Uploaded to {s3_uri}")

    conn.close()


if __name__ == "__main__":
    cli()
