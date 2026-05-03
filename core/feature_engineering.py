"""
Modeling-layer transforms — called ONLY by ML functions, never touches display df.
One-hot encodes categoricals, scales numerics, handles NaNs.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from core.schema_profiler import ColumnProfile, ColumnRole


def prepare_for_modeling(df, schema, target_col, feature_cols):
    """
    Prepare data for ML modeling.
    Returns (X: np.ndarray, y: np.ndarray, feature_names: list[str])
    """
    # Filter to relevant columns only
    cols = [c for c in feature_cols if c in df.columns and c != target_col]
    work = df[cols + [target_col]].copy()

    # Drop rows with missing target
    work = work.dropna(subset=[target_col])

    # Separate numeric and categorical features
    numeric_feats = []
    cat_feats = []
    for c in cols:
        if c in schema:
            role = schema[c].role
            if role in (ColumnRole.CATEGORY_LOW, ColumnRole.CATEGORY_HIGH,
                        ColumnRole.GEOGRAPHIC, ColumnRole.BOOLEAN):
                cat_feats.append(c)
            elif role in (ColumnRole.MONETARY, ColumnRole.QUANTITY,
                          ColumnRole.PERCENTAGE, ColumnRole.NUMERIC_OTHER):
                numeric_feats.append(c)
            # Skip IDENTIFIER, DATE, FREE_TEXT
        else:
            if pd.api.types.is_numeric_dtype(work[c]):
                numeric_feats.append(c)
            else:
                cat_feats.append(c)

    # Fill remaining NaNs
    for c in numeric_feats:
        work[c] = work[c].fillna(work[c].median())
    for c in cat_feats:
        work[c] = work[c].fillna("_missing_")

    # One-hot encode categoricals
    if cat_feats:
        work = pd.get_dummies(work, columns=cat_feats, drop_first=True, dtype=float)

    # Build X and y
    y = work[target_col].values
    feature_names = [c for c in work.columns if c != target_col]
    X = work[feature_names].values.astype(float)

    # Scale numeric features
    if len(X) > 0:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, feature_names
