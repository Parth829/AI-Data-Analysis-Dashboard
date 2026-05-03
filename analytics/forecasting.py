"""
Forecasting: naive baseline (always available) + optional Prophet.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Prophet is fully optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def naive_forecast(df, date_col, target_col, periods=30):
    """
    Naive baseline: repeat last value for future periods.
    Returns (forecast_df, metrics_on_holdout).
    """
    temp = df[[date_col, target_col]].dropna().copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna().sort_values(date_col).reset_index(drop=True)

    if len(temp) < 10:
        return None, None, "Not enough data points for forecasting."

    # Holdout: last 20% for validation
    holdout_n = max(int(len(temp) * 0.2), 1)
    train = temp.iloc[:-holdout_n]
    test = temp.iloc[-holdout_n:]

    last_val = train[target_col].iloc[-1]

    # Holdout metrics
    test_preds = np.full(len(test), last_val)
    rmse = np.sqrt(np.mean((test[target_col].values - test_preds) ** 2))
    mape = np.mean(np.abs((test[target_col].values - test_preds) / np.where(test[target_col].values == 0, 1, test[target_col].values))) * 100

    # Future forecast
    last_date = temp[date_col].iloc[-1]
    freq = pd.infer_freq(temp[date_col]) or "D"
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": last_val,
        "yhat_lower": last_val * 0.9,
        "yhat_upper": last_val * 1.1,
        "method": "naive",
    })

    metrics = {"rmse": rmse, "mape": mape, "method": "Naive (last value)"}
    return forecast_df, metrics, None


def prophet_forecast(df, date_col, target_col, periods=30):
    """
    Prophet forecast with holdout validation.
    Returns (forecast_df, metrics, components_fig, error_msg).
    """
    if not PROPHET_AVAILABLE:
        return None, None, None, "Prophet not installed."

    temp = df[[date_col, target_col]].copy()
    temp.columns = ["ds", "y"]
    temp["ds"] = pd.to_datetime(temp["ds"], errors="coerce")
    temp["y"] = pd.to_numeric(temp["y"], errors="coerce")
    temp["y"] = temp["y"].replace([np.inf, -np.inf], np.nan)
    temp = temp.dropna().sort_values("ds").reset_index(drop=True)

    if len(temp) < 10:
        return None, None, None, "Not enough valid data."
    if temp["y"].nunique() <= 1:
        return None, None, None, "Target has no variance."

    # Remove timezone
    temp["ds"] = temp["ds"].dt.tz_localize(None) if hasattr(temp["ds"].dtype, "tz") and temp["ds"].dtype.tz else temp["ds"]

    # Holdout
    holdout_n = max(int(len(temp) * 0.2), 1)
    train = temp.iloc[:-holdout_n]
    test = temp.iloc[-holdout_n:]

    try:
        m = Prophet(daily_seasonality=False)
        m.fit(train)

        # Validate on holdout
        test_future = m.make_future_dataframe(periods=holdout_n)
        test_forecast = m.predict(test_future)
        test_preds = test_forecast.iloc[-holdout_n:]["yhat"].values
        test_actual = test["y"].values

        rmse = np.sqrt(np.mean((test_actual - test_preds) ** 2))
        mape = np.mean(np.abs((test_actual - test_preds) / np.where(test_actual == 0, 1, test_actual))) * 100

        # Full model
        m_full = Prophet(daily_seasonality=False)
        m_full.fit(temp)
        future = m_full.make_future_dataframe(periods=periods)
        forecast = m_full.predict(future)

        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
        forecast_df["method"] = "Prophet"

        metrics = {"rmse": rmse, "mape": mape, "method": "Prophet"}

        # Components
        try:
            comp_fig = m_full.plot_components(forecast)
        except Exception:
            comp_fig = None

        return forecast_df, metrics, comp_fig, None

    except Exception as e:
        return None, None, None, f"Prophet error: {str(e)}"


def build_forecast_chart(df, date_col, target_col, forecast_df, title="Forecast"):
    """Build a Plotly chart showing actual data + forecast."""
    fig = go.Figure()

    # Actual data
    temp = df[[date_col, target_col]].dropna().sort_values(date_col)
    fig.add_trace(go.Scatter(
        x=temp[date_col], y=temp[target_col],
        mode="lines+markers", name="Actual",
        marker=dict(size=3, color="#1f77b4"),
        line=dict(color="#1f77b4"),
    ))

    if forecast_df is not None and len(forecast_df) > 0:
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["yhat"],
            mode="lines", name=f"Forecast ({forecast_df['method'].iloc[0]})",
            line=dict(color="#ff7f0e", dash="dash"),
        ))
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["yhat_upper"],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["yhat_lower"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(255,127,14,0.15)", showlegend=False,
        ))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=target_col, hovermode="x unified")
    return fig
