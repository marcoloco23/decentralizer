"""Claude LLM integration for graph intelligence interpretation."""

from __future__ import annotations

import anthropic
import pandas as pd

from decentralizer.config import get_settings


def get_client() -> anthropic.Anthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def interpret_community(
    community_id: int,
    members: list[str],
    stats: dict,
    top_addresses: pd.DataFrame | None = None,
) -> str:
    """Generate LLM interpretation of a community cluster."""
    client = get_client()

    context = f"""Community #{community_id} in an Ethereum transaction network:
- Size: {stats.get('size', len(members))} addresses
- Internal edges: {stats.get('internal_edges', 'unknown')}
- Internal value: {stats.get('internal_value', 'unknown'):.4f} ETH
- Density: {stats.get('density', 'unknown'):.6f}
- Sample addresses: {', '.join(members[:10])}"""

    if top_addresses is not None and not top_addresses.empty:
        context += f"\n- Top addresses by PageRank:\n{top_addresses.head(5).to_string()}"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Analyze this Ethereum transaction network community and provide a brief interpretation. What might this cluster represent? Consider DeFi protocols, exchanges, MEV bots, NFT marketplaces, or other on-chain entities.

{context}

Provide a 2-3 sentence interpretation.""",
        }],
    )
    return message.content[0].text


def summarize_address(
    address: str,
    metrics: dict,
    recent_transactions: pd.DataFrame | None = None,
) -> str:
    """Generate a summary of an address's behavior."""
    client = get_client()

    context = f"""Ethereum address: {address}
Metrics: {metrics}"""

    if recent_transactions is not None and not recent_transactions.empty:
        context += f"\nRecent transactions (last 10):\n{recent_transactions.head(10).to_string()}"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Analyze this Ethereum address based on its network metrics and transaction patterns. Classify it (e.g., whale, bot, DeFi user, exchange) and summarize its behavior in 2-3 sentences.

{context}""",
        }],
    )
    return message.content[0].text


def explain_risk_score(address: str, anomaly_score: float, metrics: dict) -> str:
    """Generate explanation for an address's risk/anomaly score."""
    client = get_client()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""An anomaly detection model assigned a score of {anomaly_score:.4f} to Ethereum address {address}.

Address metrics: {metrics}

In 1-2 sentences, explain what this anomaly score might indicate about this address's behavior. Higher scores indicate more unusual patterns.""",
        }],
    )
    return message.content[0].text


def query_graph(question: str, graph_context: str) -> str:
    """Natural language queries over graph data."""
    client = get_client()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""You are an blockchain network analyst. Answer the following question using the provided graph data context.

Graph Context:
{graph_context}

Question: {question}

Provide a clear, data-driven answer.""",
        }],
    )
    return message.content[0].text
