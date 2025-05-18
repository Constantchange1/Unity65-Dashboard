
# Unity65 Dashboard - Rebuild

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Unity65 Forecast Dashboard")
st.title("Unity65 Forecast Dashboard")

uploaded_file = st.file_uploader("Upload Unity Forecast JSON or CSV", type=["json", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df["Momentum"] = df["Predicted_Close"].diff().fillna(0)
    df["Entropy"] = -np.log(np.clip(df["Confidence"], 1e-6, 1))
    df["Phi"] = df["Resonance_Index"] * df["Confidence"]
    df["Chi"] = (df["Resonance_Index"]**2) * (df["Phi"]**1.618) / (df["Entropy"] + 1e-6)
    df["Omega_hope"] = (df["Momentum"] * df["Resonance_Index"]) / (df["Entropy"] + 1e-6)
    df["Omega_fear"] = df["Entropy"] * (1 - df["Resonance_Index"]) * (1 - df["Momentum"])
    df["Echo"] = df["Phi"].rolling(window=5).mean().fillna(0)
    df["Composite"] = df["Chi"] + df["Omega_hope"] - df["Omega_fear"] + df["Echo"]

    df["Predicted_Up"] = df["Composite"].diff().fillna(0) > 0
    df["Actual_Up"] = df["Close"].diff().fillna(0) > 0
    df["Correct"] = df["Predicted_Up"] == df["Actual_Up"]
    accuracy = df["Correct"].mean() * 100

    st.subheader("Price & Forecast Composite")
    fig = px.line(df, x=df.index, y=["Close", "Predicted_Close", "Composite"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Symbolic Fields")
    fig2 = px.line(df, x=df.index, y=["Chi", "Omega_hope", "Omega_fear", "Echo"])
    st.plotly_chart(fig2, use_container_width=True)

    st.metric("Prediction Accuracy", f"{accuracy:.2f}%")

    st.dataframe(df.tail(10))
else:
    st.info("Upload a Unity JSON or CSV file to begin.")
