"""
main.py
-------
End-to-end pipeline for Wireless Network Traffic Prediction.

Steps:
1. Load preprocessed data (from /data/processed/)
2. Train and run LightGBM, XGBoost, and LSTM models
3. Evaluate and compare their performance
4. Display and save comparison results
"""

import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Import your model modules
from src.lightgbm_model import train_lightgbm
from src.xgboost_model import train_xgboost
from src.lstm_model import train_lstm


# ======================================================
# 1. Define Data Directory
# ======================================================
DATA_DIR = r"L:\MAIN-PROJECT\data\processed"

# Check if data exists
required_files = [
    "X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy",
    "X_seq_train.npy", "y_seq_train.npy", "X_seq_test.npy", "y_seq_test.npy"
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing_files:
    raise FileNotFoundError(f"❌ Missing preprocessed files: {missing_files}\nRun preprocess_for_models.py first.")

print("✅ All preprocessed files found. Proceeding with model training...\n")


# ======================================================
# 2. Run Machine Learning Models
# ======================================================
# LightGBM
model_lgb, y_test_lgb, y_pred_lgb = train_lightgbm(DATA_DIR)

# XGBoost
model_xgb, y_test_xgb, y_pred_xgb = train_xgboost(DATA_DIR)

# LSTM
model_lstm, y_test_lstm, y_pred_lstm = train_lstm(DATA_DIR, epochs=20, batch_size=32)


# ======================================================
# 3. Model Evaluation
# ======================================================
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MSE": mse, "RMSE": rmse, "R²": r2}


results = []
results.append(evaluate_model(y_test_lgb, y_pred_lgb, "LightGBM"))
results.append(evaluate_model(y_test_xgb, y_pred_xgb, "XGBoost"))
results.append(evaluate_model(y_test_lstm, y_pred_lstm, "LSTM"))

# Convert to DataFrame
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Results:")
print(results_df)

# Save results
output_path = os.path.join(DATA_DIR, "model_comparison_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")


# ======================================================
# 4. Summary
# ======================================================

print("\n🎯 End-to-End Wireless Network Traffic Prediction Pipeline Completed Successfully ✅")
