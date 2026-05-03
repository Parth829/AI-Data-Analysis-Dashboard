"""
Forecasting page — naive baseline + optional Prophet.
"""
import streamlit as st
import pandas as pd
import numpy as np
from core.schema_profiler import get_date_columns, get_numeric_columns
from analytics.forecasting import (
    naive_forecast, prophet_forecast, build_forecast_chart, PROPHET_AVAILABLE,
)
from visualizations.chart_themes import apply_theme


def render_forecasting(df, schema):
    """Render the forecasting page."""
    st.header("Forecasting")

    date_cols = get_date_columns(schema)
    num_cols = [c for c in get_numeric_columns(schema) if c in df.columns]

    if not date_cols:
        st.warning("No date column detected. Forecasting requires a date column.")
        return
    if not num_cols:
        st.warning("No numeric columns available for forecasting.")
        return

    c1, c2 = st.columns(2)
    with c1:
        dc = st.selectbox("Date column:", date_cols, key="fc_date")
    with c2:
        target = st.selectbox("Target column:", num_cols, key="fc_target")

    periods = st.slider("Forecast periods ahead:", 7, 365, 30, key="fc_periods")

    if not PROPHET_AVAILABLE:
        st.info("Prophet not installed — using naive baseline. Install `prophet` for advanced forecasting.")

    methods = ["Naive Baseline"]
    if PROPHET_AVAILABLE:
        methods.append("Prophet")
        methods.append("Compare Both")
    method = st.radio("Method:", methods, horizontal=True, key="fc_method")

    if st.button("Run Forecast", key="run_forecast"):
        with st.spinner("Forecasting..."):
            # Naive
            naive_fc, naive_metrics, naive_err = naive_forecast(df, dc, target, periods)
            if naive_err:
                st.error(naive_err)
                return

            if method == "Naive Baseline" or (method == "Compare Both"):
                st.subheader("Naive Baseline")
                fig = build_forecast_chart(df, dc, target, naive_fc, "Naive Forecast")
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                if naive_metrics:
                    mc1, mc2 = st.columns(2)
                    mc1.metric("RMSE", f"{naive_metrics['rmse']:,.2f}")
                    mc2.metric("MAPE", f"{naive_metrics['mape']:.1f}%")

            if method in ("Prophet", "Compare Both") and PROPHET_AVAILABLE:
                st.subheader("Prophet Forecast")
                prophet_fc, prophet_metrics, comp_fig, prophet_err = prophet_forecast(df, dc, target, periods)
                if prophet_err:
                    st.error(prophet_err)
                else:
                    fig = build_forecast_chart(df, dc, target, prophet_fc, "Prophet Forecast")
                    apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    if prophet_metrics:
                        mc1, mc2 = st.columns(2)
                        mc1.metric("RMSE", f"{prophet_metrics['rmse']:,.2f}")
                        mc2.metric("MAPE", f"{prophet_metrics['mape']:.1f}%")

                    # Components
                    if comp_fig:
                        with st.expander("Trend & Seasonality Components"):
                            st.pyplot(comp_fig)

            # Comparison table
            if method == "Compare Both" and naive_metrics and prophet_metrics:
                st.markdown("---")
                st.subheader("Method Comparison")
                comp_df = pd.DataFrame([
                    {"Method": "Naive", "RMSE": naive_metrics["rmse"], "MAPE (%)": naive_metrics["mape"]},
                    {"Method": "Prophet", "RMSE": prophet_metrics["rmse"], "MAPE (%)": prophet_metrics["mape"]},
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                better = "Prophet" if prophet_metrics["rmse"] < naive_metrics["rmse"] else "Naive"
                st.success(f"**{better}** performs better on the holdout set.")

            # Forecast data
            fc_data = naive_fc if method == "Naive Baseline" else (prophet_fc if prophet_fc is not None else naive_fc)
            if fc_data is not None:
                with st.expander("View Forecast Data"):
                    display = fc_data[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                    display.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]
                    st.dataframe(display, use_container_width=True, hide_index=True)
