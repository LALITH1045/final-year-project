import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

# Import models
from src.lightgbm_model import train_lightgbm
from src.xgboost_model import train_xgboost
from src.lstm_model import train_lstm

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/engineered_6G_network_slicing_qos_dataset_2345.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    return df

df = load_data()

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Wireless Traffic Analyzer",
    page_icon="📶",
    layout="wide"
)

# Sidebar Navigation
menu = st.sidebar.radio(
    "📌 MENU",
    ["Dashboard", "Traffic Analyzer", "Model Evaluation"]
)

# --------------------------------------------------
# ⭐ DASHBOARD PAGE
# --------------------------------------------------
if menu == "Dashboard":
    st.title("📶 Wireless Network Traffic Dashboard")
    st.write("Explore traffic patterns and monitor network behavior.")

    st.subheader("📘 Sample Dataset")
    with st.expander("Show Data Preview"):
        st.dataframe(df.head(50))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Average Traffic", f"{df['Traffic Load (bps)'].mean():,.0f} bps")
    col3.metric("Max Traffic Recorded", f"{df['Traffic Load (bps)'].max():,.0f} bps")

    # Line chart
    st.subheader("📈 Traffic Trend Over Time")
    fig = px.line(df, x="Timestamp", y="Traffic Load (bps)",
                  title="Traffic Rise & Fall", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Hour-wise bar chart
    df["hour"] = df["Timestamp"].dt.hour
    hourly = df.groupby("hour")["Traffic Load (bps)"].mean().reset_index()
    st.subheader("🕒 Average Hourly Traffic")
    fig2 = px.bar(hourly, x="hour", y="Traffic Load (bps)",
                  template="plotly_dark", title="Traffic by Hour")
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# ⭐ TRAFFIC ANALYZER PAGE
# --------------------------------------------------
elif menu == "Traffic Analyzer":
    st.title("🔍 Wireless Traffic Analyzer")
    st.write("Enter a time to check traffic level from dataset patterns.")

    user_time = st.time_input("Select Time", value=dt.time(14, 0))
    hour_selected = user_time.hour

    df["hour"] = df["Timestamp"].dt.hour
    subset = df[df["hour"] == hour_selected]

    if subset.empty:
        st.warning("⚠ No traffic data available for this time.")
    else:
        avg_traffic = subset["Traffic Load (bps)"].mean()

        def classify(bps):
            if bps < 1e7: return "🟢 Low Traffic"
            elif bps < 3e7: return "🟡 Moderate Traffic"
            else: return "🔴 High Traffic"

        level = classify(avg_traffic)

        st.success(f"Traffic at {hour_selected}:00 → {level}")
        st.metric("Average Traffic", f"{avg_traffic:,.0f} bps")

        fig3 = px.line(subset, x="Timestamp", y="Traffic Load (bps)",
                       template="plotly_dark",
                       title=f"Traffic Around {hour_selected}:00")
        st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# ⭐ MODEL EVALUATION PAGE
# --------------------------------------------------
elif menu == "Model Evaluation":
    st.title("🧠 Machine Learning Model Evaluation Dashboard")
    st.write("Evaluate LightGBM, XGBoost, and LSTM models using preprocessed data.")

    data_dir = "data/processed"

    if st.button("Run All Models"):
        st.info("⏳ Training models... please wait...")

        # ---------------------------
        # LightGBM
        # ---------------------------
        model_lgb, y_test_lgb, y_pred_lgb = train_lightgbm(data_dir)

        # ---------------------------
        # XGBoost
        # ---------------------------
        model_xgb, y_test_xgb, y_pred_xgb = train_xgboost(data_dir)

        # ---------------------------
        # LSTM
        # ---------------------------
        model_lstm, y_test_lstm, y_pred_lstm = train_lstm(data_dir)

        # ---------------------------
        # Evaluation Function
        # ---------------------------
        def evaluate(y_true, y_pred, model_name):
            mse = mean_squared_error(y_true, y_pred)
            rmse = sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            return {"Model": model_name, "MSE": mse, "RMSE": rmse, "R²": r2}

        results = [
            evaluate(y_test_lgb, y_pred_lgb, "LightGBM"),
            evaluate(y_test_xgb, y_pred_xgb, "XGBoost"),
            evaluate(y_test_lstm, y_pred_lstm, "LSTM")
        ]

        results_df = pd.DataFrame(results)

        st.subheader("📊 Model Comparison Table")
        st.dataframe(results_df.style.background_gradient(cmap="Blues"))

        # Best model
        best = results_df.loc[results_df["R²"].idxmax()]
        st.success(f"🏆 Best Model: **{best['Model']}** (R² = {best['R²']:.4f})")

        # -------------------------------------
        # Predicted vs Actual Plot (LSTM)
        # -------------------------------------
        st.subheader("📈 Predicted vs Actual (Best Model)")

        actual = y_test_lstm[:200]
        pred = y_pred_lstm[:200]

        df_plot = pd.DataFrame({
            "Actual Traffic": actual,
            "Predicted Traffic": pred.flatten()
        })

        fig4 = px.line(df_plot, 
                       title="LSTM Predicted vs Actual",
                       labels={"value": "Traffic Load (bps)"})
        st.plotly_chart(fig4, use_container_width=True)
