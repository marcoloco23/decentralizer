"""Communities page - cluster visualization and LLM interpretation."""

import streamlit as st
import plotly.express as px

from decentralizer.storage.database import get_connection
from decentralizer.graph.builder import build_graph
from decentralizer.graph.communities import louvain_communities, label_propagation, community_stats
from decentralizer.graph.algorithms import page_rank


@st.cache_resource
def _get_db():
    return get_connection()


@st.cache_data(ttl=300)
def _load_graph(chain_id: int):
    conn = _get_db()
    return build_graph(conn, chain_id=chain_id, financial_only=True)


@st.cache_data(ttl=300)
def _detect_communities(chain_id: int, algorithm: str, resolution: float):
    G = _load_graph(chain_id)
    if G.number_of_nodes() == 0:
        return None, None, G
    if algorithm == "Louvain":
        comm_df = louvain_communities(G, resolution=resolution)
    else:
        comm_df = label_propagation(G)
    stats_df = community_stats(G, comm_df)
    return comm_df, stats_df, G


st.title("Community Detection")

chain_id = st.sidebar.selectbox("Chain", [1, 42161, 10, 8453, 137], format_func=lambda x: {
    1: "Ethereum", 42161: "Arbitrum", 10: "Optimism", 8453: "Base", 137: "Polygon"
}.get(x, str(x)))

algorithm = st.sidebar.selectbox("Algorithm", ["Louvain", "Label Propagation"])
resolution = st.sidebar.slider("Resolution (Louvain)", 0.1, 3.0, 1.0, step=0.1) if algorithm == "Louvain" else 1.0

comm_df, stats_df, G = _detect_communities(chain_id, algorithm, resolution)

if comm_df is None or G.number_of_nodes() == 0:
    st.warning("No graph data available.")
else:
    st.subheader("Community Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Communities Found", len(stats_df))
    c2.metric("Largest Community", int(stats_df["size"].max()) if not stats_df.empty else 0)
    c3.metric("Avg Community Size", f"{stats_df['size'].mean():.1f}" if not stats_df.empty else "0")

    # Community size distribution
    if not stats_df.empty:
        fig = px.bar(
            stats_df.head(30), x="community_id", y="size",
            title="Top 30 Communities by Size",
            labels={"size": "Members", "community_id": "Community ID"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Community Statistics")
        st.dataframe(stats_df, use_container_width=True)

    # Explore individual community
    st.subheader("Explore Community")
    if not stats_df.empty:
        selected_community = st.selectbox(
            "Select Community",
            stats_df["community_id"].tolist(),
            format_func=lambda x: f"Community {x} ({stats_df[stats_df['community_id']==x]['size'].iloc[0]} members)"
        )

        members = comm_df[comm_df["community_id"] == selected_community]["address"].tolist()
        st.write(f"**{len(members)} members**")

        # Show top addresses in community by PageRank
        subgraph = G.subgraph(members)
        pr_df = page_rank(subgraph, top_k=20)
        if not pr_df.empty:
            st.write("**Top addresses by PageRank:**")
            st.dataframe(pr_df, use_container_width=True)

        # LLM interpretation
        if st.button("Generate AI Interpretation"):
            try:
                from decentralizer.ml.llm.client import interpret_community
                cstats = stats_df[stats_df["community_id"] == selected_community].iloc[0].to_dict()
                interpretation = interpret_community(
                    selected_community, members, cstats, pr_df
                )
                st.success(interpretation)
            except Exception as e:
                st.error(f"LLM interpretation failed: {e}")
