"""
preprocess_for_models.py
------------------------

Full preprocessing pipeline for wireless traffic prediction.
Automatically:
1. Cleans raw dataset (if needed)
2. Applies feature engineering
3. Preprocesses features (encoding, scaling, splitting)
4. Creates sequences for LSTM
5. Saves all processed outputs
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Import modules
from data_cleaning import clean_dataset
from feature_engineering import feature_engineering_pipeline


# ======================================================
# 1. File Paths
# ======================================================
RAW_DATA_PATH = r"L:\MAIN-PROJECT\data\6G_network_slicing_qos_dataset_2345.xlsx"
CLEANED_DATA_PATH = r"L:\MAIN-PROJECT\data\processed_6G_network_slicing_qos_dataset_2345.csv"
ENGINEERED_DATA_PATH = r"L:\MAIN-PROJECT\data\engineered_6G_network_slicing_qos_dataset_2345.csv"
OUTPUT_DIR = r"L:\MAIN-PROJECT\data\processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================================
# 2. Step 1: Clean Data (if needed)
# ======================================================
if not os.path.exists(CLEANED_DATA_PATH):
    print("⚙️ Cleaned dataset not found. Running cleaning process...")
    clean_dataset(RAW_DATA_PATH, CLEANED_DATA_PATH)
else:
    print("✅ Cleaned dataset found.")


# ======================================================
# 3. Step 2: Run Feature Engineering (auto-clean fallback)
# ======================================================
if not os.path.exists(ENGINEERED_DATA_PATH):
    print("🧩 Feature-engineered dataset not found. Running feature engineering module...")
    df = feature_engineering_pipeline(CLEANED_DATA_PATH, ENGINEERED_DATA_PATH)
else:
    print("✅ Feature-engineered dataset found. Loading existing version...")
    df = pd.read_csv(ENGINEERED_DATA_PATH)

print(f"📄 Dataset ready for preprocessing | Shape: {df.shape}")


# ======================================================
# 4. Define Target Column
# ======================================================
TARGET = "Traffic Load (bps)"

if TARGET not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET}' not found in dataset!")

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ======================================================
# 5. Encode Categorical Columns
# ======================================================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

if categorical_cols:
    print("🔤 Encoding categorical columns:", categorical_cols)
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))
else:
    print("ℹ️ No categorical columns to encode.")


# ======================================================
# 6. Scale Numerical Features
# ======================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("✅ Features scaled successfully using MinMaxScaler.")


# ======================================================
# 7. Train-Test Split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)
print("✅ Train-test split complete.")
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# ======================================================
# 8. Create Sequences for LSTM
# ======================================================
def create_sequences(X, y, time_steps=10):
    """
    Creates time-based sequences for LSTM input.
    Each sample uses 'time_steps' past records to predict the next one.
    """
    X_seq, y_seq = [], []
    X = pd.DataFrame(X)
    y = pd.Series(y).reset_index(drop=True)

    for i in range(time_steps, len(X)):
        X_seq.append(X.iloc[i - time_steps:i].values)
        y_seq.append(y.iloc[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"✅ Created LSTM sequences: {X_seq.shape[0]} samples, {time_steps} timesteps, {X_seq.shape[2]} features.")
    return X_seq, y_seq


TIME_STEPS = 10
X_seq_train, y_seq_train = create_sequences(X_train, y_train, TIME_STEPS)
X_seq_test, y_seq_test = create_sequences(X_test, y_test, TIME_STEPS)


# ======================================================
# 9. Save Processed Data
# ======================================================
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

np.save(os.path.join(OUTPUT_DIR, "X_seq_train.npy"), X_seq_train)
np.save(os.path.join(OUTPUT_DIR, "y_seq_train.npy"), y_seq_train)
np.save(os.path.join(OUTPUT_DIR, "X_seq_test.npy"), X_seq_test)
np.save(os.path.join(OUTPUT_DIR, "y_seq_test.npy"), y_seq_test)

print("\n🎯 Full preprocessing pipeline (clean → feature engineer → preprocess → LSTM sequence) complete.")
print(f"📁 Processed data saved in: {OUTPUT_DIR}")
