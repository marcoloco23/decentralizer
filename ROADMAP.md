# Decentralizer - Roadmap & Vision

## What Is This?

Decentralizer is a **blockchain network intelligence platform** that treats Ethereum (and other EVM chains) as a social network. Instead of looking at individual transactions, it analyzes the *graph structure* of who transacts with whom to answer questions like:

- **Who are the most influential addresses?** (PageRank, influence maximization)
- **Which addresses behave similarly?** (community detection, address recommendations)
- **What are the hidden clusters?** (Louvain communities — exchanges, DeFi protocols, MEV bots, wash traders)
- **Which addresses are anomalous?** (GNN-based anomaly detection)
- **Will two addresses transact in the future?** (link prediction)

### Real-World Use Cases

| Who | What They'd Use It For |
|---|---|
| **Security researchers** | Trace fund flows from exploits/hacks, identify mixers and laundering patterns |
| **Compliance teams** | Map address clusters to identify sanctioned entity interactions |
| **DeFi analysts** | Understand protocol usage patterns, find whale behavior clusters |
| **MEV researchers** | Identify searcher networks, builder-proposer relationships |
| **Academic researchers** | Study network effects in decentralized systems |
| **Investors/traders** | Track smart money movements, find alpha in network patterns |

---

## Current State (v2.0) - What's Built

### Data Pipeline
- [x] Multi-chain EVM data fetching (Ethereum, Arbitrum, Optimism, Base, Polygon)
- [x] Async parallel block fetcher (1000 blocks in ~52s)
- [x] DuckDB local storage (750k+ transactions, indexed)
- [x] Legacy CSV migration from v1

### Graph Analysis
- [x] PageRank / Weighted PageRank / Personalized PageRank
- [x] Influence maximization (greedy)
- [x] Betweenness centrality (igraph, approximate for large graphs)
- [x] Clustering coefficients
- [x] Community detection (Louvain, Label Propagation via igraph)
- [x] Address recommendations (cosine similarity on neighbor overlap)
- [x] K-core decomposition

### Machine Learning
- [x] GraphSAGE and GAT node embeddings (PyTorch Geometric)
- [x] GNN link prediction (~79% accuracy)
- [x] Autoencoder anomaly detection
- [x] Claude LLM integration for natural language queries

### Dashboard
- [x] Address Explorer (search, metrics, transaction history)
- [x] Interactive graph visualization (pyvis)
- [x] Graph analytics (all algorithms in tabbed view)
- [x] Community detection & exploration
- [x] ML predictions page
- [x] AI chat interface

### Performance
- [x] igraph C backend for expensive algorithms (13-127x speedup)
- [x] SQL indexes and targeted queries (no full table scans)
- [x] Cached algorithm results in dashboard (5min TTL)

---

## Roadmap

### Phase 1: Data Quality & Scale

**Goal**: Go from "demo with 1000 blocks" to "production with millions of transactions"

- [ ] **Incremental fetching** — Track last fetched block per chain, only fetch new blocks. Currently refetches from latest every time.
- [ ] **Historical backfill** — Fetch specific block ranges (e.g., "all blocks from Jan 2024"). Add `--start-block` / `--end-block` CLI flags.
- [ ] **Contract labeling** — Integrate Etherscan/4byte verified contract labels. Know that `0xdAC1...` is "USDT" not just an address hash.
- [ ] **Token transfers** — Parse ERC-20 Transfer events from transaction logs. Currently only captures native ETH value, missing the majority of DeFi activity.
- [ ] **Internal transactions** — Fetch trace data for contract-to-contract calls. These are invisible in normal transactions but represent significant fund flows.
- [ ] **Multi-chain correlation** — Detect the same entity operating across chains (bridge activity, cross-chain arbitrage).

### Phase 2: Smarter Graph Analysis

**Goal**: Move from "run algorithms" to "surface actionable insights automatically"

- [ ] **Temporal graphs** — Track how the network evolves over time. Which communities are growing? Which addresses became central recently?
- [ ] **Motif detection** — Find recurring transaction patterns (triangular trades, hub-and-spoke, chains). These reveal wash trading, layered money laundering, and MEV strategies.
- [ ] **Flow analysis** — Max-flow / min-cut between addresses. "What's the maximum value that could flow from A to B through the network?"
- [ ] **Address classification** — Auto-label addresses as: EOA, DEX, CEX, bridge, mixer, lending protocol, NFT marketplace, etc. using transaction pattern heuristics.
- [ ] **Risk scoring** — Composite risk score combining anomaly detection, proximity to known bad actors, transaction pattern irregularities, and mixer usage.

### Phase 3: Better ML

**Goal**: Beat the 79% link prediction baseline and add more predictive capabilities

- [ ] **Richer node features** — Current features are just degree/value stats. Add: gas usage patterns, time-of-day activity, contract interaction diversity, account age.
- [ ] **Temporal GNN** — Use TGN (Temporal Graph Networks) to capture time-dependent patterns. An address that suddenly changes behavior is more interesting than one that's always been weird.
- [ ] **Subgraph-level anomaly detection** — Don't just flag individual addresses. Flag suspicious *patterns* (e.g., a ring of 5 addresses cycling funds).
- [ ] **Transfer learning across chains** — Train on Ethereum, fine-tune on Arbitrum. Network patterns should be similar across EVM chains.
- [ ] **Explainable predictions** — For any flagged address, show *why* it was flagged: "High anomaly score because: (1) unusual in-degree spike, (2) all counterparties are <7 days old, (3) 94% of value flows to a single address."

### Phase 4: Real-Time & Monitoring

**Goal**: Move from "batch analysis" to "live monitoring"

- [ ] **WebSocket block streaming** — Subscribe to new blocks in real-time instead of polling.
- [ ] **Incremental graph updates** — Add new edges to the graph without rebuilding from scratch.
- [ ] **Alert system** — "Notify me when address X receives > 100 ETH" or "Alert when a new address enters community #42."
- [ ] **Live dashboard** — Auto-refresh dashboard as new blocks come in. Show real-time transaction flow animation.

### Phase 5: API & Integration

**Goal**: Make Decentralizer useful as infrastructure, not just a dashboard

- [ ] **REST API** — FastAPI endpoints: `/address/{addr}/metrics`, `/graph/communities`, `/predict/link?from=X&to=Y`, `/risk/{addr}`
- [ ] **Export formats** — GraphML, GEXF, CSV, JSON export for use in Gephi, Neo4j, or other tools.
- [ ] **Webhook integrations** — Push alerts to Slack, Discord, Telegram, email.
- [ ] **Notebook integration** — Jupyter kernel extension for interactive analysis with the Decentralizer graph.

### Phase 6: UX & Visualization

**Goal**: Make the dashboard genuinely useful for non-technical users

- [ ] **Better graph viz** — Replace pyvis with deck.gl or Sigma.js for GPU-accelerated rendering of large graphs (current limit: ~2000 edges).
- [ ] **Address profiles** — Rich profile pages with activity timeline, community membership, risk breakdown, similar addresses.
- [ ] **Investigation mode** — Click an address, expand its neighborhood, trace fund flows interactively. Like a blockchain Maltego.
- [ ] **Saved queries & reports** — Save analysis configurations, generate PDF reports.
- [ ] **Dark pool detection dashboard** — Dedicated view for identifying suspicious activity clusters.

---

## Architecture

```
                    ┌─────────────┐
                    │  EVM Chains  │
                    │ ETH/ARB/OP  │
                    └──────┬──────┘
                           │ web3.py async
                    ┌──────▼──────┐
                    │   Fetcher    │
                    │  (parallel)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   DuckDB     │
                    │  (local DB)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼───┐ ┌──────▼──────┐
       │  NetworkX /  │ │ PyG  │ │   Claude    │
       │   igraph     │ │ GNN  │ │    LLM      │
       │  (analysis)  │ │ (ML) │ │  (explain)  │
       └──────┬───────┘ └──┬───┘ └──────┬──────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │  Streamlit   │
                    │  Dashboard   │
                    └─────────────┘
```

## Quick Start

```bash
# Install
uv sync

# Import existing data
uv run decentralizer migrate

# Fetch live data from Ethereum
uv run decentralizer fetch --chain ethereum --blocks 1000

# Run graph analysis
uv run decentralizer analyze

# Train ML models
uv run decentralizer train

# Launch dashboard
uv run decentralizer dashboard
```

---

## Contributing

This is a research/portfolio project. Key areas where contributions would be valuable:

1. **Token transfer parsing** — ERC-20 event log decoding
2. **Graph visualization** — GPU-accelerated large graph rendering
3. **Address labeling** — Integrating public label databases
4. **Temporal analysis** — Time-windowed graph metrics
