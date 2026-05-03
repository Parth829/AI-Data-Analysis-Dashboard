"""
Regression with proper evaluation: R², MAE, MAPE, RMSE, residuals, cross-validation.
Supports categorical features via one-hot encoding inside this module.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from ml.model_evaluation import compute_regression_metrics, plot_actual_vs_predicted, plot_residuals


def run_regression(df, target_col, feature_cols, schema=None):
    """
    Run Random Forest regression with full evaluation.
    Returns dict with keys: train_metrics, test_metrics, cv_metrics,
    fig_actual_pred, fig_residuals, importances, error_msg.
    """
    result = {"error_msg": None}

    # Prepare data — handle categoricals internally
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        result["error_msg"] = "No valid feature columns selected."
        return result

    work = df[cols + [target_col]].dropna(subset=[target_col]).copy()

    # Separate numeric vs categorical
    cat_cols = work[cols].select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = work[cols].select_dtypes(include=[np.number]).columns.tolist()

    # Fill missing
    for c in num_cols:
        work[c] = work[c].fillna(work[c].median())
    for c in cat_cols:
        work[c] = work[c].fillna("_missing_")

    # One-hot encode
    if cat_cols:
        work = pd.get_dummies(work, columns=cat_cols, drop_first=True, dtype=float)

    feat_names = [c for c in work.columns if c != target_col]
    X = work[feat_names].values.astype(float)
    y = work[target_col].values.astype(float)

    if len(X) < 10:
        result["error_msg"] = "Not enough data (need at least 10 rows after cleaning)."
        return result

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    result["train_metrics"] = compute_regression_metrics(y_train, train_pred)
    result["test_metrics"] = compute_regression_metrics(y_test, test_pred)

    # Plots
    result["fig_actual_pred"] = plot_actual_vs_predicted(y_test, test_pred, target_col)
    result["fig_residuals"] = plot_residuals(y_test, test_pred, target_col)

    # Feature importances
    result["importances"] = pd.DataFrame({
        "Feature": feat_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # Cross-validation (5-fold)
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = []
        for train_idx, val_idx in kf.split(X):
            m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            m.fit(X[train_idx], y[train_idx])
            pred = m.predict(X[val_idx])
            cv_metrics.append(compute_regression_metrics(y[val_idx], pred))

        # Average
        avg = {}
        for key in cv_metrics[0]:
            vals = [m[key] for m in cv_metrics]
            avg[key] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
        result["cv_metrics"] = cv_metrics
        result["cv_summary"] = avg
    except Exception:
        result["cv_metrics"] = None
        result["cv_summary"] = None

    return result
