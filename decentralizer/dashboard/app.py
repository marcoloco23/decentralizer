"""Streamlit multi-page application entry point."""

import streamlit as st


def main():
    st.set_page_config(
        page_title="Decentralizer",
        page_icon="🔗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = {
        "Address Explorer": [
            st.Page("decentralizer/dashboard/pages/explorer.py", title="Address Explorer", icon="🔍"),
        ],
        "Graph Analysis": [
            st.Page("decentralizer/dashboard/pages/graph_viz.py", title="Graph Visualization", icon="🕸️"),
            st.Page("decentralizer/dashboard/pages/analytics.py", title="Graph Analytics", icon="📊"),
            st.Page("decentralizer/dashboard/pages/communities.py", title="Communities", icon="👥"),
        ],
        "Machine Learning": [
            st.Page("decentralizer/dashboard/pages/ml.py", title="ML Predictions", icon="🤖"),
            st.Page("decentralizer/dashboard/pages/chat.py", title="AI Chat", icon="💬"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
