"""
feature_engineering.py
-----------------------

Enhanced feature engineering module with automatic data cleaning.
If the cleaned dataset does not exist, it triggers the cleaning process first.

This ensures smooth pipeline flow:
Raw Data → Cleaned Data → Feature Engineered Data
"""

import pandas as pd
import numpy as np
import os
from data_cleaning import clean_dataset  # import cleaning function


# =====================================================
# 1. Load Cleaned Dataset (with auto-clean fallback)
# =====================================================
def load_cleaned_data(path, raw_data_path):
    """
    Loads the cleaned dataset. If missing, runs data cleaning automatically.
    """
    if not os.path.exists(path):
        print(f"⚠️ Cleaned dataset not found at {path}. Running cleaning process first...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clean_dataset(raw_data_path, path)

    df = pd.read_csv(path)
    print(f"✅ Loaded cleaned dataset from {path} | Shape: {df.shape}")
    return df


# =====================================================
# 2. Extract Time-based Features
# =====================================================
def create_time_features(df, time_col="Timestamp"):
    if time_col not in df.columns:
        raise KeyError(f"❌ Timestamp column '{time_col}' not found in dataset.")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    df["hour"] = df[time_col].dt.hour
    df["day"] = df[time_col].dt.day
    df["weekday"] = df[time_col].dt.weekday
    df["month"] = df[time_col].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    print("🕒 Time-based features extracted: hour, day, weekday, month, is_weekend")
    return df


# =====================================================
# 3. Create Statistical / Derived Features
# =====================================================
def create_statistical_features(df):
    cols = df.columns

    if "Traffic Load (bps)" in cols and "Bandwidth Utilization (%)" in cols:
        df["traffic_to_bandwidth_ratio"] = df["Traffic Load (bps)"] / (df["Bandwidth Utilization (%)"] + 1e-5)

    if "Signal Strength (dBm)" in cols and "Latency (ms)" in cols:
        df["signal_latency_interaction"] = df["Signal Strength (dBm)"] * df["Latency (ms)"]

    if "Network Utilization (%)" in cols:
        df["network_utilization_squared"] = df["Network Utilization (%)"] ** 2

    if "Packet Loss Rate (%)" in cols:
        df["log_packet_loss"] = np.log1p(df["Packet Loss Rate (%)"])

    print("📈 Derived features created:")
    print("    ➤ traffic_to_bandwidth_ratio")
    print("    ➤ signal_latency_interaction")
    print("    ➤ network_utilization_squared")
    print("    ➤ log_packet_loss")
    return df


# =====================================================
# 4. Encode Categorical Features
# =====================================================
def encode_categorical(df):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        print("ℹ️ No categorical columns to encode.")
        return df

    for col in categorical_cols:
        if df[col].nunique() <= 10:
            df = pd.get_dummies(df, columns=[col], prefix=col)
        else:
            df[col] = df[col].astype("category").cat.codes

    print(f"🔤 Encoded categorical columns: {categorical_cols}")
    return df


# =====================================================
# 5. Save Feature-Engineered Dataset
# =====================================================
def save_engineered_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Feature-engineered dataset saved to: {output_path}")


# =====================================================
# 6. Feature Engineering Pipeline
# =====================================================
def feature_engineering_pipeline(input_path, output_path, time_col="Timestamp"):
    """
    Full pipeline: ensures cleaned dataset exists → applies feature engineering.
    """
    raw_data_path = "data/6G_network_slicing_qos_dataset_2345.xlsx"

    # Step 1: Load (auto-clean if needed)
    df = load_cleaned_data(input_path, raw_data_path)

    # Step 2: Time-based features
    df = create_time_features(df, time_col)

    # Step 3: Derived statistical features
    df = create_statistical_features(df)

    # Step 4: Encode categorical variables
    df = encode_categorical(df)

    # Step 5: Save result
    save_engineered_data(df, output_path)

    print("🎯 Feature engineering process complete.")
    return df


# =====================================================
# 7. Run as Script
# =====================================================
if __name__ == "__main__":
    INPUT_PATH = r"L:\MAIN-PROJECT\data\processed_6G_network_slicing_qos_dataset_2345.csv"
    OUTPUT_PATH = r"L:\MAIN-PROJECT\data\engineered_6G_network_slicing_qos_dataset_2345.csv"

    df_engineered = feature_engineering_pipeline(INPUT_PATH, OUTPUT_PATH)
    print("\nPreview of Engineered Data:")
    print(df_engineered.head())
