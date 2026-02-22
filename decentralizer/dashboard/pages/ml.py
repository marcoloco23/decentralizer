"""ML Predictions page - link prediction, anomaly scores, GNN embeddings."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from decentralizer.config import get_settings, PROJECT_ROOT


st.title("ML Predictions")

model_dir = PROJECT_ROOT / "models"

st.sidebar.subheader("Model Settings")
model_type = st.sidebar.selectbox("GNN Model", ["graphsage", "gat"])

tab1, tab2, tab3 = st.tabs(["Link Prediction", "Anomaly Detection", "Train Model"])

with tab1:
    st.subheader("Link Prediction")
    st.write("Predict the probability of a transaction link between two addresses.")

    col1, col2 = st.columns(2)
    addr1 = col1.text_input("Address 1", key="lp_addr1")
    addr2 = col2.text_input("Address 2", key="lp_addr2")

    if st.button("Predict Link") and addr1 and addr2:
        model_path = model_dir / f"{model_type}_link_predictor.pt"
        if not model_path.exists():
            st.warning(f"No trained model found at {model_path}. Train a model first using the 'Train Model' tab or `decentralizer train`.")
        else:
            try:
                import torch
                from decentralizer.storage.database import get_connection
                from decentralizer.graph.builder import build_graph
                from decentralizer.ml.gnn.embeddings import graph_to_pyg_data
                from decentralizer.ml.gnn.trainer import create_model, load_model

                conn = get_connection()
                G = build_graph(conn, chain_id=1)
                data = graph_to_pyg_data(G)

                if addr1.lower() not in data.node_to_idx or addr2.lower() not in data.node_to_idx:
                    st.error("One or both addresses not found in the graph.")
                else:
                    model = create_model(model_type, in_channels=data.x.size(1))
                    model = load_model(model, model_path)
                    model.eval()

                    with torch.no_grad():
                        z = model.encode(data.x, data.edge_index)
                        idx1 = data.node_to_idx[addr1.lower()]
                        idx2 = data.node_to_idx[addr2.lower()]
                        edge = torch.tensor([[idx1], [idx2]], dtype=torch.long)
                        prob = torch.sigmoid(model.predict(z, edge)).item()

                    st.metric("Link Probability", f"{prob:.4f}")
                    if prob > 0.8:
                        st.success("High probability of a transaction link.")
                    elif prob > 0.5:
                        st.info("Moderate probability of a transaction link.")
                    else:
                        st.warning("Low probability of a transaction link.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Anomaly Detection")
    st.write("Addresses ranked by anomaly score (higher = more unusual behavior).")

    anomaly_path = model_dir / "anomaly_scores.csv"
    if anomaly_path.exists():
        anom_df = pd.read_csv(anomaly_path)
        st.dataframe(anom_df.head(100), use_container_width=True)

        fig = px.histogram(anom_df, x="anomaly_score", nbins=50,
                          title="Anomaly Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomaly scores computed yet. Run `decentralizer train` to generate them.")

with tab3:
    st.subheader("Train GNN Model")
    st.write("Train a GNN model on the current graph data.")

    epochs = st.slider("Epochs", 50, 500, 200)
    lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.01, format="%.4f")

    if st.button("Start Training"):
        with st.spinner("Training model... This may take a while."):
            try:
                import torch
                from decentralizer.storage.database import get_connection
                from decentralizer.graph.builder import build_graph
                from decentralizer.ml.gnn.embeddings import graph_to_pyg_data, prepare_link_prediction_data
                from decentralizer.ml.gnn.trainer import create_model, train_link_prediction, train_anomaly_detector, save_model

                conn = get_connection()
                G = build_graph(conn, chain_id=1, financial_only=True)
                data = graph_to_pyg_data(G)
                split = prepare_link_prediction_data(data)

                model = create_model(model_type, in_channels=data.x.size(1))
                device = "mps" if torch.backends.mps.is_available() else "cpu"

                history = train_link_prediction(model, split, epochs=epochs, lr=lr, device=device)

                # Save model
                model_dir.mkdir(parents=True, exist_ok=True)
                save_model(model, model_dir / f"{model_type}_link_predictor.pt")

                metrics = history["test_metrics"]
                st.success(f"Training complete! Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, MCC: {metrics['mcc']:.4f}")

                # Train anomaly detector
                model.eval()
                with torch.no_grad():
                    z = model.encode(data.x.to(device), split["train_pos"].to(device))
                ae, scores = train_anomaly_detector(z, embedding_dim=z.size(1), device=device)

                # Save anomaly scores
                anom_df = pd.DataFrame({
                    "address": data.node_ids,
                    "anomaly_score": scores.numpy(),
                })
                anom_df = anom_df.sort_values("anomaly_score", ascending=False)
                anom_df.to_csv(model_dir / "anomaly_scores.csv", index=False)
                save_model(ae, model_dir / "anomaly_autoencoder.pt")

                st.info(f"Anomaly scores saved for {len(anom_df)} addresses.")

            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())
