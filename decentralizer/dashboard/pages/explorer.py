"""Address Explorer page - search addresses, view metrics and transactions."""

import streamlit as st
import pandas as pd
import plotly.express as px

from decentralizer.storage.database import get_connection, get_transactions, get_address_metrics
from decentralizer.graph.algorithms import address_degree
from decentralizer.graph.builder import build_graph


@st.cache_resource
def _get_db():
    return get_connection()


@st.cache_data(ttl=300)
def _load_graph(chain_id: int):
    conn = _get_db()
    return build_graph(conn, chain_id=chain_id)


st.title("Address Explorer")

chain_id = st.sidebar.selectbox("Chain", [1, 42161, 10, 8453, 137], format_func=lambda x: {
    1: "Ethereum", 42161: "Arbitrum", 10: "Optimism", 8453: "Base", 137: "Polygon"
}.get(x, str(x)))

address = st.text_input("Search Address", placeholder="0x...")

if address:
    address = address.lower().strip()
    conn = _get_db()

    # Get transactions for this address
    all_txs = get_transactions(conn, chain_id=chain_id)
    addr_txs = all_txs[(all_txs["sender"] == address) | (all_txs["receiver"] == address)]

    if addr_txs.empty:
        st.warning("No transactions found for this address on this chain.")
    else:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        sent = addr_txs[addr_txs["sender"] == address]
        received = addr_txs[addr_txs["receiver"] == address]

        col1.metric("Transactions", len(addr_txs))
        col2.metric("Sent", len(sent))
        col3.metric("Received", len(received))
        col4.metric("Total Value (ETH)", f"{addr_txs['value'].sum():.4f}")

        # Graph metrics
        G = _load_graph(chain_id)
        deg_info = address_degree(G, address)
        st.subheader("Network Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("In-Degree", deg_info["in_degree"])
        m2.metric("Out-Degree", deg_info["out_degree"])
        m3.metric("Total Degree", deg_info["total_degree"])

        # Address metrics from DB
        metrics_df = get_address_metrics(conn, chain_id=chain_id)
        if not metrics_df.empty:
            addr_metrics = metrics_df[metrics_df["address"] == address]
            if not addr_metrics.empty:
                row = addr_metrics.iloc[0]
                st.subheader("Computed Metrics")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("PageRank", f"{row.get('page_rank', 0):.6f}")
                c2.metric("Betweenness", f"{row.get('betweenness_centrality', 0):.6f}")
                c3.metric("Community", int(row.get("community_id", -1)))
                c4.metric("Anomaly Score", f"{row.get('anomaly_score', 0):.4f}")

        # Transaction history
        st.subheader("Transaction History")
        display_df = addr_txs[["hash", "block_number", "sender", "receiver", "value", "timestamp", "gas"]].copy()
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], unit="s")
        st.dataframe(display_df.sort_values("timestamp", ascending=False), use_container_width=True)

        # Value over time chart
        if len(addr_txs) > 1:
            chart_df = addr_txs.copy()
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], unit="s")
            chart_df["direction"] = chart_df.apply(
                lambda r: "Sent" if r["sender"] == address else "Received", axis=1
            )
            fig = px.scatter(
                chart_df, x="timestamp", y="value", color="direction",
                title="Transaction Values Over Time",
                labels={"value": "Value (ETH)", "timestamp": "Time"},
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter an Ethereum address to explore its transaction history and network metrics.")

    # Show overview stats
    conn = _get_db()
    all_txs = get_transactions(conn, chain_id=chain_id)
    if not all_txs.empty:
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", len(all_txs))
        c2.metric("Unique Senders", all_txs["sender"].nunique())
        c3.metric("Unique Receivers", all_txs["receiver"].nunique())
