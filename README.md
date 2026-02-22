# Decentralizer v2.0

A multi-chain blockchain intelligence platform with GNN + LLM capabilities. Analyzes Ethereum and EVM-compatible chain transaction networks to identify influential addresses, detect communities, predict links, and detect anomalies.

## Architecture

```
decentralizer/
├── config.py           # pydantic-settings configuration (.env)
├── models/schema.py    # Pydantic v2 data models with chain_id
├── chain/              # Multi-chain async data collection (web3.py 7.x)
├── storage/            # DuckDB storage + Parquet export
├── graph/              # NetworkX/igraph graph algorithms
├── ml/
│   ├── gnn/            # GraphSAGE, GAT, link prediction, anomaly detection
│   └── llm/            # Claude integration for interpretation
├── dashboard/          # Streamlit multi-page app
└── cli.py              # Click CLI
```

## Quick Start

```bash
# Install dependencies (using uv)
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (Infura, Etherscan, Anthropic)

# Import legacy CSV data into DuckDB
uv run decentralizer migrate

# Or fetch fresh data from a chain
uv run decentralizer fetch --chain ethereum --blocks 100

# Run graph analysis
uv run decentralizer analyze

# Train GNN models
uv run decentralizer train --model graphsage --epochs 200

# Launch the dashboard
uv run decentralizer dashboard
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `decentralizer migrate` | Import legacy CSVs into DuckDB |
| `decentralizer fetch --chain <name> --blocks <n>` | Fetch blocks from an EVM chain |
| `decentralizer analyze --chain-id <id>` | Run all graph algorithms |
| `decentralizer train --model <type>` | Train GNN models (graphsage/gat) |
| `decentralizer dashboard` | Launch Streamlit web app |

## Supported Chains

| Chain | ID | Name |
|-------|----|------|
| Ethereum | 1 | ethereum |
| Arbitrum | 42161 | arbitrum |
| Optimism | 10 | optimism |
| Base | 8453 | base |
| Polygon | 137 | polygon |

## Graph Algorithms

- **PageRank** — address importance by transaction flow
- **Weighted PageRank** — weighted by ETH value
- **Personalized PageRank** — biased toward seed addresses
- **Influence Maximization** — greedy max-reach selection
- **Address Recommendations** — cosine similarity on neighbor overlap
- **Betweenness Centrality** — bridge node detection
- **Clustering Coefficients** — local clustering
- **K-Core Decomposition** — dense subgraph identification
- **Louvain Communities** — community detection
- **Label Propagation** — fast community detection

## ML Models

- **GraphSAGE / GAT** — inductive node embeddings via GNN
- **Link Prediction** — predict transaction probability between addresses
- **Anomaly Detection** — autoencoder on embeddings to flag unusual addresses
- **LLM Integration** — Claude-powered community interpretation, address summarization, and natural language graph queries

## Dashboard Pages

1. **Address Explorer** — search any address, view metrics and transaction history
2. **Graph Visualization** — interactive pyvis network graph
3. **Graph Analytics** — PageRank, influence, centrality, recommendations
4. **Communities** — cluster detection with AI interpretation
5. **ML Predictions** — link prediction, anomaly scores, model training
6. **AI Chat** — natural language queries over graph data

## Tech Stack

- **Data**: DuckDB, Parquet, Pandas
- **Blockchain**: web3.py 7.x (async), multi-chain EVM
- **Graph**: NetworkX, python-igraph
- **ML**: PyTorch, PyTorch Geometric (GraphSAGE, GAT)
- **LLM**: Anthropic Claude API
- **UI**: Streamlit, Pyvis, Plotly
- **Config**: pydantic-settings, python-dotenv

## Migration from v1

v1 used MongoDB Atlas, TigerGraph Cloud, and Graphistry — all external services with hardcoded credentials. v2 is fully local-first:

- MongoDB → DuckDB (embedded, zero config)
- TigerGraph GSQL → NetworkX/igraph (pure Python)
- Graphistry → Pyvis (local rendering)
- Node2Vec + XGBoost → GraphSAGE/GAT + link predictor
- Hardcoded secrets → `.env` file

**Note**: Legacy credentials in git history from `src/utils/constants.py` should be rotated.
