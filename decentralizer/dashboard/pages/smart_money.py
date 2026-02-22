"""Smart Money Leaderboard — top wallets by composite score with drill-down."""

import streamlit as st
import plotly.graph_objects as go

from decentralizer.storage.database import (
    get_connection,
    get_smart_money_leaderboard,
    get_smart_money_score,
    get_wallet_pnl,
    get_dex_trades_for_address,
    get_token_transfers_for_address,
)

CHAIN_NAMES = {1: "Ethereum", 42161: "Arbitrum", 10: "Optimism", 8453: "Base", 137: "Polygon"}


@st.cache_resource
def _get_db():
    return get_connection()


@st.cache_data(ttl=300)
def _load_leaderboard(chain_id: int, top_k: int):
    conn = _get_db()
    return get_smart_money_leaderboard(conn, chain_id=chain_id, top_k=top_k)


@st.cache_data(ttl=300)
def _load_wallet_detail(chain_id: int, address: str):
    conn = _get_db()
    score = get_smart_money_score(conn, address, chain_id)
    pnl = get_wallet_pnl(conn, address, chain_id)
    trades = get_dex_trades_for_address(conn, address, chain_id, limit=50)
    transfers = get_token_transfers_for_address(conn, address, chain_id, limit=50)
    return score, pnl, trades, transfers


st.title("Smart Money Leaderboard")

chain_id = st.sidebar.selectbox(
    "Chain",
    [1, 42161, 10, 8453, 137],
    format_func=lambda x: CHAIN_NAMES.get(x, str(x)),
)
top_k = st.sidebar.slider("Top K", 10, 500, 100)

leaderboard = _load_leaderboard(chain_id, top_k)

if leaderboard.empty:
    st.warning("No smart money scores found. Run `decentralizer score --chain-id {}` first.".format(chain_id))
    st.stop()

# Leaderboard table
st.subheader(f"Top {len(leaderboard)} Wallets by Composite Score")

display_cols = ["rank", "address", "composite_score", "pnl_score", "page_rank_score", "early_entry_score", "concentration_score"]
available_cols = [c for c in display_cols if c in leaderboard.columns]

st.dataframe(
    leaderboard[available_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "address": st.column_config.TextColumn("Address", width="large"),
        "composite_score": st.column_config.NumberColumn("Composite", format="%.4f"),
        "pnl_score": st.column_config.NumberColumn("P&L Score", format="%.4f"),
        "page_rank_score": st.column_config.NumberColumn("PageRank", format="%.4f"),
        "early_entry_score": st.column_config.NumberColumn("Early Entry", format="%.4f"),
        "concentration_score": st.column_config.NumberColumn("Diversification", format="%.4f"),
    },
)

# Score distribution
st.subheader("Score Distribution")
col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(data=[go.Histogram(x=leaderboard["composite_score"], nbinsx=30)])
    fig.update_layout(title="Composite Score Distribution", xaxis_title="Score", yaxis_title="Count", height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if "pnl_score" in leaderboard.columns:
        fig2 = go.Figure(data=[go.Histogram(x=leaderboard["pnl_score"], nbinsx=30)])
        fig2.update_layout(title="P&L Score Distribution", xaxis_title="Score", yaxis_title="Count", height=300)
        st.plotly_chart(fig2, use_container_width=True)

# Drill-down
st.subheader("Wallet Drill-Down")
selected_address = st.text_input(
    "Enter wallet address to inspect",
    value=leaderboard.iloc[0]["address"] if not leaderboard.empty else "",
)

if selected_address:
    score, pnl, trades, transfers = _load_wallet_detail(chain_id, selected_address.lower())

    if score:
        st.markdown("#### Score Breakdown")
        cols = st.columns(5)
        cols[0].metric("Composite", f"{score.get('composite_score', 0):.4f}")
        cols[1].metric("P&L", f"{score.get('pnl_score', 0):.4f}")
        cols[2].metric("PageRank", f"{score.get('page_rank_score', 0):.4f}")
        cols[3].metric("Early Entry", f"{score.get('early_entry_score', 0):.4f}")
        cols[4].metric("Diversification", f"{score.get('concentration_score', 0):.4f}")

        # Radar chart
        categories = ["PageRank", "P&L", "Early Entry", "Diversification"]
        values = [
            score.get("page_rank_score", 0),
            score.get("pnl_score", 0),
            score.get("early_entry_score", 0),
            score.get("concentration_score", 0),
        ]
        values.append(values[0])  # Close the polygon
        categories.append(categories[0])

        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill="toself"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=350, title="Score Radar")
        st.plotly_chart(fig, use_container_width=True)

    if not pnl.empty:
        st.markdown("#### Token Holdings & P&L")
        st.dataframe(pnl, use_container_width=True, hide_index=True)

    if not trades.empty:
        st.markdown("#### Recent DEX Trades")
        st.dataframe(trades, use_container_width=True, hide_index=True)

    if not transfers.empty:
        st.markdown("#### Recent Token Transfers")
        st.dataframe(transfers.head(20), use_container_width=True, hide_index=True)
