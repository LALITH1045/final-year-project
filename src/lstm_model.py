"""
lstm_model.py
--------------
Trains an LSTM model using sequential data 
and returns the trained model along with predictions.
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_lstm(data_dir, epochs=20, batch_size=32):
    print("\n🚀 Running LSTM model...")

    # Load sequence data
    X_seq_train = np.load(os.path.join(data_dir, "X_seq_train.npy"))
    y_seq_train = np.load(os.path.join(data_dir, "y_seq_train.npy"))
    X_seq_test  = np.load(os.path.join(data_dir, "X_seq_test.npy"))
    y_seq_test  = np.load(os.path.join(data_dir, "y_seq_test.npy"))

    input_shape = (X_seq_train.shape[1], X_seq_train.shape[2])
    print(f"✅ LSTM Input Shape: {input_shape}")

    # Build LSTM architecture
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train model
    model.fit(
        X_seq_train, y_seq_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    # Predict
    y_pred = model.predict(X_seq_test)

    print("✅ LSTM training complete.")
    return model, y_seq_test, y_pred
