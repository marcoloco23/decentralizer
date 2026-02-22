# Decentralizer - Roadmap & Vision

## What Is This?

Decentralizer is an **on-chain intelligence platform for traders and investors**. It maps the hidden network structure of blockchain transactions to give you an edge that price charts alone can't provide.

Every token swap, every whale transfer, every fund movement creates a connection in a massive graph. Decentralizer analyzes that graph to answer the questions that matter:

- **Where is smart money flowing right now?** — Track wallets with the highest network influence across DeFi
- **Which wallets are accumulating before a move?** — Detect coordinated buying patterns across wallet clusters
- **Who are the whales connected to?** — Map the transaction network around any address to understand who they trade with
- **Is this a real trend or wash trading?** — Community detection separates organic activity from artificial volume
- **What will this wallet do next?** — GNN-based link prediction forecasts future transaction connections

### The Edge

Most trading tools show you *what* happened (price, volume, TVL). Decentralizer shows you *who* is behind it and *how* they're connected. A token pumping 50% means nothing without context. That same pump driven by a cluster of wallets that preceded 3 previous 10x plays — that's signal.

| Traditional Tools | Decentralizer |
|---|---|
| Price charts, candlesticks | Who is buying and their track record |
| Total volume | Real vs. wash volume (community detection) |
| Whale alert: "100 ETH moved" | Where it came from, where it's going, and the full network around it |
| Token holder count | Holder *network* — are they connected or independent? |
| "Smart money" labels (manual) | Algorithmic smart money scoring via PageRank + influence analysis |

---

## Current State (v2.0) - What's Built

### Data Pipeline
- [x] Multi-chain EVM data fetching (Ethereum, Arbitrum, Optimism, Base, Polygon)
- [x] Async parallel block fetcher (1000 blocks in ~52s)
- [x] DuckDB local storage (750k+ transactions, indexed)
- [x] Legacy CSV migration from v1

### Graph Analysis
- [x] PageRank / Weighted PageRank — rank wallets by network influence
- [x] Influence maximization — find wallets that reach the most other wallets
- [x] Betweenness centrality — find bridge wallets connecting separate clusters
- [x] Community detection (Louvain, Label Propagation) — group wallets into clusters
- [x] Address recommendations — "wallets similar to this one"
- [x] K-core decomposition — find the densely connected core of the network

### Machine Learning
- [x] GraphSAGE and GAT node embeddings (PyTorch Geometric)
- [x] GNN link prediction (~79% accuracy)
- [x] Autoencoder anomaly detection
- [x] Claude LLM integration for natural language queries

### Dashboard
- [x] Address Explorer — search any wallet, view metrics and transaction history
- [x] Interactive graph visualization (pyvis)
- [x] Graph analytics with all algorithms
- [x] Community detection & exploration
- [x] ML predictions page
- [x] AI chat — ask questions in plain English

### Performance
- [x] igraph C backend (13-127x speedup over pure Python)
- [x] SQL indexes and targeted queries
- [x] Cached results in dashboard

---

## Roadmap

### Phase 1: Smart Money Tracking

**Goal**: Identify and follow the wallets that consistently make profitable trades

- [ ] **ERC-20 token transfers** — Parse Transfer events from transaction logs. Currently only tracks native ETH. This is the single biggest gap — most DeFi alpha is in token movements, not raw ETH.
- [ ] **DEX trade reconstruction** — Decode Uniswap/Sushiswap/Curve swap events to know exactly what tokens a wallet bought and sold, at what price, and what their P&L was.
- [ ] **Wallet P&L scoring** — Calculate realized + unrealized P&L per wallet. The wallets with the best track records are the ones worth following.
- [ ] **Smart money index** — Composite score: PageRank (network influence) + P&L track record + early entry frequency + portfolio concentration. Rank the top 1000 wallets across each chain.
- [ ] **Wallet watchlists** — Save wallets to a personal watchlist. See their recent activity, current holdings, and network position at a glance.
- [ ] **Contract labeling** — Integrate Etherscan verified labels + community-maintained tag databases. Turn `0xdAC1...` into "Tether USDT" and `0x7a25...` into "Binance Hot Wallet 14."

### Phase 2: Alpha Signals

**Goal**: Turn network analysis into actionable trade signals

- [ ] **Accumulation detection** — Flag tokens where a cluster of high-PageRank wallets are buying simultaneously. If 5 historically profitable wallets all buy the same token within 48 hours, that's a signal.
- [ ] **Fund flow tracking** — Trace large movements end-to-end. When a whale moves $10M from Binance to a DEX, what do they buy? When they move tokens to a new wallet, where does it go next?
- [ ] **Pre-listing detection** — Identify tokens receiving unusual smart money inflows before CEX listings or major announcements.
- [ ] **Wash trading filter** — Use community detection to identify circular trading patterns. Filter them out of volume metrics to see real demand.
- [ ] **Correlation signals** — "Wallets that bought token X also bought token Y within 7 days." Find basket trades and thematic plays.
- [ ] **Temporal analysis** — Track how wallet clusters evolve week over week. A growing community around a new protocol is bullish. A shrinking one is a warning sign.

### Phase 3: Real-Time Monitoring

**Goal**: Get signals as they happen, not hours later

- [ ] **WebSocket block streaming** — Subscribe to new blocks in real-time instead of batch fetching.
- [ ] **Live smart money feed** — Real-time stream: "Wallet 0xabc (PageRank #47, +340% 90d P&L) just bought 50 ETH worth of TOKEN on Uniswap."
- [ ] **Alert system** — Configurable alerts: "Notify me when any top-100 PageRank wallet buys token X" or "Alert when 3+ wallets from cluster #42 make the same trade."
- [ ] **Telegram / Discord bot** — Push alerts to where traders already are. Format: wallet label, action, amount, context (their P&L history, what cluster they belong to).
- [ ] **Incremental graph updates** — Add new edges without rebuilding the full graph. Keep PageRank and community assignments current within seconds of a new block.

### Phase 4: Portfolio Intelligence

**Goal**: Help traders make better decisions about their own positions

- [ ] **Portfolio X-ray** — Input your own wallets. See which smart money communities you overlap with, which you don't. "You hold 60% of what Cluster #7 (DeFi yield farmers) holds. You're missing: TOKEN_A, TOKEN_B."
- [ ] **Exit signal detection** — "3 of the 5 top wallets that bought TOKEN_X before its last pump have started selling this week."
- [ ] **Counterparty analysis** — For any trade you're considering: who's on the other side? Are smart money wallets selling what you're buying?
- [ ] **Network-based risk score** — A token held by 1000 unconnected wallets is safer than one held by 1000 wallets that are all controlled by the same cluster.
- [ ] **Cross-chain tracking** — Detect the same entity operating across Ethereum, Arbitrum, Base. Follow smart money as they rotate between L1 and L2s.

### Phase 5: Product & Distribution

**Goal**: Ship something traders will actually pay for

- [ ] **REST API** — FastAPI endpoints for programmatic access. Let quant traders integrate signals into their own systems. Endpoints: `/smart-money/top`, `/wallet/{addr}/score`, `/signals/accumulation`, `/token/{addr}/holder-network`.
- [ ] **Better visualization** — Replace pyvis with GPU-accelerated rendering (deck.gl or Sigma.js) for interactive exploration of large wallet networks. Click a whale, expand their network, trace flows visually.
- [ ] **Wallet profiles** — Rich profile pages: P&L history, current holdings, network position, community membership, similar wallets, activity timeline.
- [ ] **Signal backtesting** — "If I had followed the top 50 PageRank wallets' buys over the last 6 months, what would my return be?" Validate the alpha before trading on it.
- [ ] **Mobile-friendly dashboard** — Responsive layout. Traders check signals from their phone.
- [ ] **Saved screens & alerts dashboard** — Personal workspace with saved queries, active alerts, watchlist performance.

### Phase 6: Competitive Moat

**Goal**: Build defensible advantages that free tools can't replicate

- [ ] **Proprietary wallet labels** — Build the most comprehensive wallet labeling database through a combination of on-chain heuristics, community contributions, and ML classification.
- [ ] **Temporal GNNs** — Train models on time-series graph data. Predict which wallets will become influential *before* they do. Early detection of emerging smart money.
- [ ] **Multi-chain graph fusion** — Build a unified graph across all EVM chains. Most tools are single-chain. Cross-chain intelligence is rare and valuable.
- [ ] **LLM-powered research assistant** — "Tell me everything about this wallet cluster and what they've been doing this week" — with full context from the graph, not just a ChatGPT wrapper.
- [ ] **Signal marketplace** — Let power users publish and monetize their own signal strategies built on Decentralizer's data.

---

## Why This Wins

The on-chain analytics space is crowded (Nansen, Arkham, Dune, DeBank), but most tools are:

1. **Dashboard-first** — They show data, not signals. Traders have to manually interpret charts and tables.
2. **Address-first** — They analyze individual wallets. They don't show the *network* between wallets, which is where the real alpha lives.
3. **Backward-looking** — They tell you what happened. Graph ML can predict what will happen next.

Decentralizer's edge is **graph intelligence**: treating the blockchain as a network, not a ledger. Community detection, influence propagation, and GNN-based prediction are structurally different from what Dune dashboards or Nansen labels provide.

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

## Priority Order

| Priority | Phase | Why First |
|---|---|---|
| **NOW** | Phase 1: Smart Money Tracking | Can't do anything useful without token transfer data and wallet P&L |
| **Next** | Phase 2: Alpha Signals | This is the product — without signals, it's just a research tool |
| **Then** | Phase 3: Real-Time | Stale signals are worthless. Traders need speed |
| **Then** | Phase 4: Portfolio Intelligence | Deepens engagement — makes it personal to each user |
| **Later** | Phase 5: Product & Distribution | API + UX polish for growth |
| **Long-term** | Phase 6: Competitive Moat | Defensibility through data + ML advantages |
