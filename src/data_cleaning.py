"""
data_cleaning.py
----------------

This module cleans the wireless network dataset for machine learning
and deep learning traffic prediction.

Steps:
1. Load raw dataset (Excel/CSV)
2. Detect and parse timestamp column
3. Remove duplicates and invalid entries
4. Handle missing values (numeric + categorical)
5. Detect and handle outliers using IQR method
6. Save the cleaned dataset for preprocessing and model training
"""

import pandas as pd
import numpy as np
import os

# =====================================================
# 1. Load the dataset
# =====================================================
def load_dataset(file_path):
    """
    Loads a dataset (Excel or CSV) into a pandas DataFrame.
    """
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide CSV or XLSX.")
    print(f"✅ Dataset loaded successfully from: {file_path}")
    print(f"Shape: {df.shape}")
    return df


# =====================================================
# 2. Detect and parse timestamp column
# =====================================================
def parse_timestamp(df):
    """
    Automatically detects and parses the timestamp column.
    """
    time_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["time", "timestamp", "date"]):
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df = df.sort_values(by=time_col).reset_index(drop=True)
        print(f"🕒 Timestamp column detected and parsed: {time_col}")
    else:
        print("⚠️ No timestamp column found. Proceeding without time-based sorting.")
    return df, time_col


# =====================================================
# 3. Handle missing values
# =====================================================
def handle_missing_values(df, time_col=None):
    """
    Fills missing values using time-aware interpolation for numeric
    and mode imputation for categorical columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric columns
    for col in numeric_cols:
        if time_col:
            # Temporarily set datetime column as index
            df = df.set_index(time_col)
            df[col] = df[col].ffill(limit=3)
            df[col] = df[col].bfill(limit=3)

            # Perform time-based interpolation safely
            df[col] = df[col].interpolate(method="time")

            # Reset the index back to normal
            df = df.reset_index()
        else:
            # No timestamp column → use generic interpolation
            df[col] = df[col].interpolate()

        # Fill any remaining NaN with median
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Categorical columns
    for col in cat_cols:
        mode_val = df[col].mode(dropna=True)
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna("Unknown")

    print("🧩 Missing values handled (interpolated + filled).")
    return df


# =====================================================
# 4. Handle outliers
# =====================================================
def handle_outliers(df):
    """
    Detects and clips outliers in numeric columns using the IQR method.
    Replaces extreme values with the column median.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        # Detect outliers
        outliers = (df[col] < lower_limit) | (df[col] > upper_limit)
        num_outliers = outliers.sum()

        # Replace outliers with median
        median_val = df[col].median()
        df.loc[outliers, col] = median_val

        if num_outliers > 0:
            print(f"⚙️ Outliers replaced in '{col}': {num_outliers} values")

    print("📉 Outlier handling complete.")
    return df


# =====================================================
# 5. Save cleaned dataset
# =====================================================
def save_cleaned_dataset(df, output_path):
    """
    Saves the cleaned dataset as a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")


# =====================================================
# 6. Main Cleaning Function
# =====================================================
def clean_dataset(input_path, output_path):
    """
    Full cleaning pipeline for the dataset.
    """
    df = load_dataset(input_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df, time_col = parse_timestamp(df)
    df = handle_missing_values(df, time_col)
    df = handle_outliers(df)
    save_cleaned_dataset(df, output_path)
    print("🎯 Dataset cleaning process complete.")
    return df


# =====================================================
# 7. Run the script (if executed directly)
# =====================================================
if __name__ == "__main__":
    INPUT_FILE = "L:/MAIN-PROJECT/data/6G_network_slicing_qos_dataset_2345.csv"
    OUTPUT_FILE = "L:/MAIN-PROJECT/data/processed/processed_6G_network_slicing_qos_dataset_2345.csv"

    cleaned_df = clean_dataset(INPUT_FILE, OUTPUT_FILE)
    print("\nPreview of cleaned data:")
    print(cleaned_df.head())
