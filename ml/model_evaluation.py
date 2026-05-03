"""
Shared ML evaluation metrics and plot generators.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def compute_regression_metrics(y_true, y_pred):
    """Compute R², MAE, MAPE, RMSE."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    # MAPE — guard against zero
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = float("inf")
    return {"R²": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def plot_actual_vs_predicted(y_true, y_pred, target_name="Target"):
    """Scatter plot of actual vs predicted with identity line."""
    fig = px.scatter(x=y_true, y=y_pred,
                     labels={"x": "Actual", "y": "Predicted"},
                     title=f"Actual vs Predicted — {target_name}")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    fig.add_shape(type="line", line=dict(dash="dash", color="gray"),
                  x0=mn, y0=mn, x1=mx, y1=mx)
    fig.update_layout(height=450)
    return fig


def plot_residuals(y_true, y_pred, target_name="Target"):
    """Residuals vs predicted values."""
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals,
                     labels={"x": "Predicted", "y": "Residual"},
                     title=f"Residual Plot — {target_name}")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(height=400)
    return fig


def plot_cv_results(cv_metrics):
    """Bar chart of cross-validation fold metrics."""
    import pandas as pd
    rows = []
    for i, m in enumerate(cv_metrics):
        for metric, val in m.items():
            rows.append({"Fold": f"Fold {i+1}", "Metric": metric, "Value": val})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Fold", y="Value", color="Metric", barmode="group",
                 title="Cross-Validation Results (5-Fold)")
    fig.update_layout(height=400)
    return fig
