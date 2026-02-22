"""Address Explorer page - search addresses, view metrics and transactions."""

import streamlit as st
import pandas as pd
import plotly.express as px

from decentralizer.storage.database import (
    get_connection, get_transactions_for_address,
    get_address_summary, get_address_metrics, get_overview_stats,
)
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

    # SQL-level filtering — no full table scan
    summary = get_address_summary(conn, address, chain_id=chain_id)

    if not summary:
        st.warning("No transactions found for this address on this chain.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions", summary["tx_count"])
        col2.metric("Sent", summary["sent_count"])
        col3.metric("Received", summary["recv_count"])
        col4.metric("Total Value (ETH)", f"{summary['total_value']:.4f}")

        # Graph metrics
        G = _load_graph(chain_id)
        deg_info = address_degree(G, address)
        st.subheader("Network Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("In-Degree", deg_info["in_degree"])
        m2.metric("Out-Degree", deg_info["out_degree"])
        m3.metric("Total Degree", deg_info["total_degree"])

        # Address metrics from DB — query single address
        metrics_df = get_address_metrics(conn, chain_id=chain_id, address=address)
        if not metrics_df.empty:
            row = metrics_df.iloc[0]
            st.subheader("Computed Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PageRank", f"{row.get('page_rank', 0):.6f}")
            c2.metric("Betweenness", f"{row.get('betweenness_centrality', 0):.6f}")
            c3.metric("Community", int(row.get("community_id", -1)))
            c4.metric("Anomaly Score", f"{row.get('anomaly_score', 0):.4f}")

        # Transaction history — indexed query with limit
        st.subheader("Transaction History")
        addr_txs = get_transactions_for_address(conn, address, chain_id=chain_id, limit=1000)
        if not addr_txs.empty:
            display_df = addr_txs.copy()
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], unit="s")
            st.dataframe(display_df.sort_values("timestamp", ascending=False), use_container_width=True)

            # Value over time chart
            if len(addr_txs) > 1:
                chart_df = addr_txs.copy()
                chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], unit="s")
                # Vectorized direction assignment
                chart_df["direction"] = "Received"
                chart_df.loc[chart_df["sender"] == address, "direction"] = "Sent"
                fig = px.scatter(
                    chart_df, x="timestamp", y="value", color="direction",
                    title="Transaction Values Over Time",
                    labels={"value": "Value (ETH)", "timestamp": "Time"},
                )
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter an Ethereum address to explore its transaction history and network metrics.")

    # Overview stats via SQL aggregation — no full table scan
    conn = _get_db()
    overview = get_overview_stats(conn, chain_id=chain_id)
    if overview["tx_count"] > 0:
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", f"{overview['tx_count']:,}")
        c2.metric("Unique Senders", f"{overview['unique_senders']:,}")
        c3.metric("Unique Receivers", f"{overview['unique_receivers']:,}")
