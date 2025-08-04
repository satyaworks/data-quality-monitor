# app.py - Data Quality Monitor using IsolationForest

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="🧪 Data Quality Monitor", layout="wide")
st.title("🧠 Data Quality Monitor with Anomaly Detection")

# Upload CSV
df = None
uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Select column for anomaly detection
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        col_to_check = st.selectbox("Select numeric column to check for anomalies:", numeric_cols)
        contamination = st.slider("Contamination (expected % of anomalies):", 0.01, 0.20, 0.05, 0.01)

        # IsolationForest
        st.info("Running IsolationForest...")
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        df = df.dropna(subset=[col_to_check])
        df['anomaly'] = model.fit_predict(df[[col_to_check]])
        df['anomaly'] = df['anomaly'].map({1: False, -1: True})

        # Plot
        fig = px.scatter(
            df, x=df.index, y=col_to_check,
            color=df['anomaly'].map({True: 'Anomaly', False: 'Normal'}),
            title="Anomaly Detection Result",
            labels={"color": "Status"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show anomaly table
        st.subheader("🚨 Detected Anomalies")
        anomalies = df[df['anomaly'] == True]
        st.dataframe(anomalies)

        # Optional: download
        csv = anomalies.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Anomalies as CSV", csv, file_name="anomalies.csv", mime="text/csv")
    else:
        st.warning("No numeric columns found in the uploaded file.")
else:
    st.info("Upload a CSV file to begin.")
