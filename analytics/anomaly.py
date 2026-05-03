"""
Anomaly detection: Z-score, IQR, Isolation Forest, time-aware rolling window.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from io import BytesIO


def detect_zscore(df, column, threshold=3.0):
    """Z-score anomaly detection. Returns boolean Series."""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series(False, index=df.index)
    mean, std = df[column].mean(), df[column].std()
    if std == 0:
        return pd.Series(False, index=df.index)
    z = np.abs((df[column] - mean) / std)
    return z > threshold


def detect_iqr(df, column, multiplier=1.5):
    """IQR anomaly detection. Returns boolean Series."""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series(False, index=df.index)
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=df.index)
    return (df[column] < q1 - multiplier * iqr) | (df[column] > q3 + multiplier * iqr)


def detect_isolation_forest(df, columns, contamination=0.05):
    """Isolation Forest multi-column detection. Returns boolean Series."""
    valid = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid:
        return pd.Series(False, index=df.index)
    X = df[valid].fillna(df[valid].median())
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    return pd.Series(preds == -1, index=df.index)


def detect_rolling_window(df, date_col, value_col, window_days=30, threshold_std=2.5):
    """
    Time-aware anomaly detection — flag values outside rolling mean ± threshold*std.
    Returns boolean Series.
    """
    if date_col not in df.columns or value_col not in df.columns:
        return pd.Series(False, index=df.index)

    temp = df[[date_col, value_col]].copy()
    temp = temp.sort_values(date_col).reset_index(drop=True)

    if not pd.api.types.is_datetime64_any_dtype(temp[date_col]):
        return pd.Series(False, index=df.index)

    # Set date as index for rolling
    temp = temp.set_index(date_col)
    rolling_mean = temp[value_col].rolling(f"{window_days}D", min_periods=5).mean()
    rolling_std = temp[value_col].rolling(f"{window_days}D", min_periods=5).std()

    upper = rolling_mean + threshold_std * rolling_std
    lower = rolling_mean - threshold_std * rolling_std

    anomalies = (temp[value_col] < lower) | (temp[value_col] > upper)
    # Map back to original index
    result = pd.Series(False, index=df.index)
    result.iloc[temp.index if isinstance(temp.index, pd.RangeIndex) else range(len(temp))] = anomalies.values
    return result


def get_context_rows(df, anomaly_mask, value_col, n_context=3):
    """For each anomaly, find N similar normal rows for comparison."""
    anomalies = df[anomaly_mask]
    normals = df[~anomaly_mask]
    if len(normals) == 0 or len(anomalies) == 0:
        return pd.DataFrame()

    context_rows = []
    for _, anom_row in anomalies.head(10).iterrows():  # limit to 10 anomalies
        if value_col in anom_row.index and pd.api.types.is_numeric_dtype(df[value_col]):
            diffs = (normals[value_col] - anom_row[value_col]).abs()
            closest = diffs.nsmallest(n_context).index
            for idx in closest:
                row = normals.loc[idx].copy()
                row["_comparison_to"] = f"anomaly at index {anom_row.name}"
                context_rows.append(row)

    return pd.DataFrame(context_rows) if context_rows else pd.DataFrame()


def export_anomalies_csv(df_anomalies):
    """Export anomaly rows as CSV bytes for download."""
    output = BytesIO()
    df_anomalies.to_csv(output, index=False)
    return output.getvalue()
