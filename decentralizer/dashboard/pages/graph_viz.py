"""Graph Visualization page - interactive pyvis graph."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis.network import Network

from decentralizer.storage.database import get_connection, get_edge_dataframe


@st.cache_resource
def _get_db():
    return get_connection()


st.title("Graph Visualization")

chain_id = st.sidebar.selectbox("Chain", [1, 42161, 10, 8453, 137], format_func=lambda x: {
    1: "Ethereum", 42161: "Arbitrum", 10: "Optimism", 8453: "Base", 137: "Polygon"
}.get(x, str(x)))

tx_type = st.sidebar.selectbox("Transaction Type", [
    "All Transactions",
    "Financial Only (value > 0)",
    "Non-Financial Only (value = 0)",
])

cutoff = st.sidebar.slider("Max Edges", min_value=100, max_value=50000, value=2000, step=100)

conn = _get_db()
financial_only = tx_type == "Financial Only (value > 0)"
df = get_edge_dataframe(conn, chain_id=chain_id, financial_only=financial_only)

if tx_type == "Non-Financial Only (value = 0)":
    df = df[df["value"] == 0]

if df.empty:
    st.warning("No transaction data available. Run `decentralizer migrate` first.")
else:
    df = df.head(cutoff)

    st.info(f"Showing {len(df)} edges with {df['sender'].nunique() + df['receiver'].nunique()} unique addresses")

    # Build pyvis network
    net = Network(height="700px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=100)

    # Add edges (pyvis auto-creates nodes)
    for _, row in df.iterrows():
        net.add_node(row["sender"], label=row["sender"][:10] + "...", title=row["sender"])
        net.add_node(row["receiver"], label=row["receiver"][:10] + "...", title=row["receiver"])
        title = f"Value: {row['value']:.4f} ETH\nBlock: {row['block_number']}"
        net.add_edge(row["sender"], row["receiver"], title=title, value=max(row["value"], 0.1))

    # Render
    html = net.generate_html()
    components.html(html, height=720, scrolling=True)
