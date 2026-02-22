"""Graph Visualization page - interactive pyvis graph. Optimized for large datasets."""

import streamlit as st
import streamlit.components.v1 as components

from decentralizer.storage.database import get_connection, get_edge_dataframe


@st.cache_resource
def _get_db():
    return get_connection()


@st.cache_data(ttl=300)
def _build_graph_html(chain_id: int, financial_only: bool, non_financial: bool, cutoff: int) -> str | None:
    """Build pyvis HTML with SQL-level limiting. Cached to avoid recomputation."""
    from pyvis.network import Network

    conn = _get_db()
    df = get_edge_dataframe(conn, chain_id=chain_id, financial_only=financial_only, limit=cutoff)

    if non_financial:
        df = df[df["value"] == 0]

    if df.empty:
        return None

    net = Network(height="700px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=100)

    # Vectorized: collect unique nodes first, then add in bulk
    senders = df["sender"].values
    receivers = df["receiver"].values
    values = df["value"].values
    block_numbers = df["block_number"].values

    seen = set()
    for s in senders:
        if s not in seen:
            seen.add(s)
            net.add_node(s, label=s[:10] + "...", title=s)
    for r in receivers:
        if r not in seen:
            seen.add(r)
            net.add_node(r, label=r[:10] + "...", title=r)

    # Add edges
    for i in range(len(df)):
        net.add_edge(
            senders[i], receivers[i],
            title=f"Value: {values[i]:.4f} ETH\nBlock: {block_numbers[i]}",
            value=max(values[i], 0.1),
        )

    return net.generate_html()


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

financial_only = tx_type == "Financial Only (value > 0)"
non_financial = tx_type == "Non-Financial Only (value = 0)"

with st.spinner("Building graph visualization..."):
    html = _build_graph_html(chain_id, financial_only, non_financial, cutoff)

if html is None:
    st.warning("No transaction data available. Run `decentralizer migrate` first.")
else:
    components.html(html, height=720, scrolling=True)
