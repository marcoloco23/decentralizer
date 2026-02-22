"""Streamlit multi-page application entry point."""

from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).parent / "pages"


def main():
    st.set_page_config(
        page_title="Decentralizer",
        page_icon="🔗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = {
        "Address Explorer": [
            st.Page(str(PAGES_DIR / "explorer.py"), title="Address Explorer", icon="🔍"),
        ],
        "Graph Analysis": [
            st.Page(str(PAGES_DIR / "graph_viz.py"), title="Graph Visualization", icon="🕸️"),
            st.Page(str(PAGES_DIR / "analytics.py"), title="Graph Analytics", icon="📊"),
            st.Page(str(PAGES_DIR / "communities.py"), title="Communities", icon="👥"),
        ],
        "Machine Learning": [
            st.Page(str(PAGES_DIR / "ml.py"), title="ML Predictions", icon="🤖"),
            st.Page(str(PAGES_DIR / "chat.py"), title="AI Chat", icon="💬"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
