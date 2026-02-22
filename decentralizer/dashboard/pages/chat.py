"""AI Chat page - natural language queries over graph data."""

import streamlit as st
import pandas as pd

from decentralizer.storage.database import get_connection, get_transaction_count, get_unique_addresses
from decentralizer.graph.builder import build_graph, graph_stats
from decentralizer.graph.algorithms import page_rank


@st.cache_resource
def _get_db():
    return get_connection()


st.title("AI Chat")
st.write("Ask natural language questions about the blockchain transaction network.")

chain_id = st.sidebar.selectbox("Chain", [1, 42161, 10, 8453, 137], format_func=lambda x: {
    1: "Ethereum", 42161: "Arbitrum", 10: "Optimism", 8453: "Base", 137: "Polygon"
}.get(x, str(x)))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the network..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                conn = _get_db()
                G = build_graph(conn, chain_id=chain_id, financial_only=True)
                stats = graph_stats(G)

                # Build context for LLM
                pr_df = page_rank(G, top_k=10)
                chain_names = {1: "Ethereum", 42161: "Arbitrum", 10: "Optimism", 8453: "Base", 137: "Polygon"}

                context = f"""Network: {chain_names.get(chain_id, f'Chain {chain_id}')}
Graph Statistics:
- Nodes: {stats['nodes']:,}
- Edges: {stats['edges']:,}
- Density: {stats['density']:.6f}
- Weakly connected components: {stats['weakly_connected_components']}

Top 10 addresses by PageRank:
{pr_df.to_string() if not pr_df.empty else 'No data available'}

Total transactions in database: {get_transaction_count(conn, chain_id)}
"""

                from decentralizer.ml.llm.client import query_graph
                response = query_graph(prompt, context)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except ValueError as e:
                if "ANTHROPIC_API_KEY" in str(e):
                    msg = "Please set ANTHROPIC_API_KEY in your .env file to use AI Chat."
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                else:
                    raise
            except Exception as e:
                msg = f"Error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
