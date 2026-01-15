"""
lightgbm_model.py
-----------------
Trains a LightGBM regression model using preprocessed data 
and returns the trained model along with predictions.
"""

import os
import numpy as np
from lightgbm import LGBMRegressor

def train_lightgbm(data_dir):
    print("\n🚀 Running LightGBM model...")

    # Load preprocessed data
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    # Train model
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("✅ LightGBM training complete.")
    return model, y_test, y_pred
