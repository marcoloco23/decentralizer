"""Graph Analytics page - PageRank, influence, centrality, recommendations."""

import streamlit as st
import plotly.express as px

from decentralizer.storage.database import get_connection
from decentralizer.graph.builder import build_graph
from decentralizer.graph import algorithms as algo


@st.cache_resource
def _get_db():
    return get_connection()


@st.cache_data(ttl=300)
def _load_graph(chain_id: int, financial_only: bool):
    conn = _get_db()
    return build_graph(conn, chain_id=chain_id, financial_only=financial_only)


@st.cache_data(ttl=300)
def _page_rank(chain_id: int, financial_only: bool, top_k: int):
    G = _load_graph(chain_id, financial_only)
    return algo.page_rank(G, top_k=top_k)


@st.cache_data(ttl=300)
def _weighted_page_rank(chain_id: int, financial_only: bool, top_k: int):
    G = _load_graph(chain_id, financial_only)
    return algo.weighted_page_rank(G, top_k=top_k)


@st.cache_data(ttl=300)
def _betweenness(chain_id: int, financial_only: bool, top_k: int, use_approx: bool):
    G = _load_graph(chain_id, financial_only)
    k_sample = min(500, G.number_of_nodes()) if use_approx else None
    return algo.betweenness_centrality(G, top_k=top_k, k_sample=k_sample)


@st.cache_data(ttl=300)
def _clustering(chain_id: int, financial_only: bool, top_k: int):
    G = _load_graph(chain_id, financial_only)
    return algo.clustering_coefficients(G, top_k=top_k)


st.title("Graph Analytics")

chain_id = st.sidebar.selectbox("Chain", [1, 42161, 10, 8453, 137], format_func=lambda x: {
    1: "Ethereum", 42161: "Arbitrum", 10: "Optimism", 8453: "Base", 137: "Polygon"
}.get(x, str(x)))

financial_only = st.sidebar.checkbox("Financial transactions only", value=True)
top_k = st.sidebar.slider("Top K results", 10, 500, 100)

G = _load_graph(chain_id, financial_only)

if G.number_of_nodes() == 0:
    st.warning("No graph data. Run `decentralizer migrate` to import data.")
else:
    from decentralizer.graph.builder import graph_stats
    stats = graph_stats(G)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{stats['nodes']:,}")
    c2.metric("Edges", f"{stats['edges']:,}")
    c3.metric("Density", f"{stats['density']:.6f}")
    c4.metric("Components", stats["weakly_connected_components"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "PageRank", "Weighted PageRank", "Influence", "Centrality", "Recommendations"
    ])

    with tab1:
        st.subheader("PageRank")
        pr_df = _page_rank(chain_id, financial_only, top_k)
        st.dataframe(pr_df, use_container_width=True)
        if not pr_df.empty and "page_rank_pct" in pr_df.columns:
            fig = px.bar(pr_df.head(20), x="address", y="page_rank_pct",
                         title="Top 20 by PageRank", labels={"page_rank_pct": "PageRank %"})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Weighted PageRank")
        wpr_df = _weighted_page_rank(chain_id, financial_only, top_k)
        st.dataframe(wpr_df, use_container_width=True)

    with tab3:
        st.subheader("Influence Maximization")
        influence_k = st.slider("Top K for influence", 5, 50, 20, key="influence_k")
        with st.spinner("Computing Influence..."):
            inf_df = algo.max_influence(G, top_k=influence_k)
        st.dataframe(inf_df, use_container_width=True)

    with tab4:
        st.subheader("Betweenness Centrality")
        use_approx = st.checkbox("Use approximate (faster)", value=G.number_of_nodes() > 10000)
        bc_df = _betweenness(chain_id, financial_only, top_k, use_approx)
        st.dataframe(bc_df, use_container_width=True)

        st.subheader("Clustering Coefficients")
        cc_df = _clustering(chain_id, financial_only, top_k)
        st.dataframe(cc_df, use_container_width=True)

    with tab5:
        st.subheader("Address Recommendations")
        source = st.text_input("Source address for recommendations", key="rec_source")
        if source:
            source = source.lower().strip()
            with st.spinner("Finding recommendations..."):
                rec_df = algo.recommend_addresses(G, source, top_k=top_k)
            if rec_df.empty:
                st.info("No recommendations found for this address.")
            else:
                st.dataframe(rec_df, use_container_width=True)

        st.subheader("Personalized PageRank")
        ppr_input = st.text_area("Source addresses (one per line)", key="ppr_input")
        if ppr_input:
            addrs = [a.strip().lower() for a in ppr_input.strip().split("\n") if a.strip()]
            with st.spinner("Computing Personalized PageRank..."):
                ppr_df = algo.personalized_page_rank(G, addrs, top_k=top_k)
            st.dataframe(ppr_df, use_container_width=True)
