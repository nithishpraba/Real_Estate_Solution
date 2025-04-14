import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data_for_training(df):
    """
    Preprocess the Real Estate DataFrame for training.
    
    - Drops missing values.
    - Expects a target column named "price" (all lowercase).
    - Splits into features (all columns except "price") and target.
    - Scales the features using StandardScaler.
    """
    df_clean = df.dropna().copy()
    target_column = "price"
    if target_column not in df_clean.columns:
        raise Exception(f"Target column '{target_column}' not found. Available columns: {df_clean.columns.tolist()}")
    
    X = df_clean.drop(target_column, axis=1)
    y = df_clean[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def preprocess_data_for_inference(df):
    """
    Preprocess the Real Estate DataFrame for inference.
    
    - Drops missing values.
    - Scales the features using StandardScaler.
    
    NOTE: In production, you typically save and re-use the scaler fitted during training.
    Here we re-fit the scaler for simplicity.
    """
    df_clean = df.dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    return X_scaled
