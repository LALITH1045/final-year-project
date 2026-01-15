"""
xgboost_model.py
----------------
Trains an XGBoost regression model using preprocessed data 
and returns the trained model along with predictions.
"""

import os
import numpy as np
from xgboost import XGBRegressor

def train_xgboost(data_dir):
    print("\n🚀 Running XGBoost model...")

    # Load preprocessed data
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    # Train model
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("✅ XGBoost training complete.")
    return model, y_test, y_pred
