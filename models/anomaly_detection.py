import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies_zscore(df, column, threshold=3):
    """
    Detects anomalies in a specific numeric column using the Z-score method.
    Anomalies are points that are more than `threshold` standard deviations from the mean.
    Returns: A boolean Series where True indicates an anomaly.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series(False, index=df.index)
        
    mean = df[column].mean()
    std = df[column].std()
    
    if std == 0:
         return pd.Series(False, index=df.index)
         
    z_scores = np.abs((df[column] - mean) / std)
    return z_scores > threshold

def detect_anomalies_isolation_forest(df, columns, contamination=0.05):
    """
    Detects anomalies across multiple numeric columns using Isolation Forest.
    Returns: A boolean Series where True indicates an anomaly.
    """
    # Filter for numeric columns that actually exist
    numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
         return pd.Series(False, index=df.index)
         
    # Handle NaNs: drop rows with NaNs in these columns temporarily for the model
    # (Though typically our data is already cleaned)
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Initialize and fit model
    model = IsolationForest(contamination=contamination, random_state=42)
    # The predict returns 1 for normal, -1 for anomaly
    predictions = model.fit_predict(X)
    
    # Convert to boolean: True if anomaly (-1)
    return predictions == -1
