"""
Anomaly detection page — Z-score, IQR, Isolation Forest, rolling window.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from core.schema_profiler import get_date_columns, get_numeric_columns
from core.data_cleaner import get_display_columns
from analytics.anomaly import (
    detect_zscore, detect_iqr, detect_isolation_forest,
    detect_rolling_window, get_context_rows, export_anomalies_csv,
)
from visualizations.chart_themes import apply_theme


def render_anomaly_detection(df, schema):
    """Render anomaly detection page."""
    st.header("Anomaly Detection")
    display_cols = get_display_columns(df)
    num_cols = [c for c in get_numeric_columns(schema) if c in df.columns]

    if not num_cols:
        st.warning("No numeric columns available for anomaly detection.")
        return

    method = st.radio("Detection method:",
                      ["Z-Score", "IQR", "Isolation Forest", "Rolling Window (time-aware)"],
                      horizontal=True, key="anom_method")

    if method == "Z-Score":
        col = st.selectbox("Column:", num_cols, key="anom_zscore_col")
        threshold = st.slider("Z-Score threshold:", 2.0, 5.0, 3.0, 0.1, key="zscore_thresh")
        if st.button("Detect Anomalies", key="run_zscore"):
            mask = detect_zscore(df, col, threshold)
            _show_results(df, mask, col, display_cols)

    elif method == "IQR":
        col = st.selectbox("Column:", num_cols, key="anom_iqr_col")
        mult = st.slider("IQR multiplier:", 1.0, 3.0, 1.5, 0.1, key="iqr_mult")
        if st.button("Detect Anomalies", key="run_iqr"):
            mask = detect_iqr(df, col, mult)
            _show_results(df, mask, col, display_cols)

    elif method == "Isolation Forest":
        sel_cols = st.multiselect("Columns:", num_cols,
                                  default=num_cols[:2] if len(num_cols) >= 2 else num_cols,
                                  key="if_cols")
        contam = st.slider("Contamination:", 0.01, 0.20, 0.05, 0.01, key="if_contam")
        if st.button("Detect Anomalies", key="run_if"):
            if not sel_cols:
                st.warning("Select at least one column.")
            else:
                mask = detect_isolation_forest(df, sel_cols, contam)
                _show_results(df, mask, sel_cols[0], display_cols)

    elif method == "Rolling Window (time-aware)":
        date_cols = get_date_columns(schema)
        if not date_cols:
            st.warning("No date column detected. Use another method.")
            return
        dc = st.selectbox("Date column:", date_cols, key="rw_date")
        val = st.selectbox("Value column:", num_cols, key="rw_val")
        window = st.slider("Window (days):", 7, 90, 30, key="rw_window")
        if st.button("Detect Anomalies", key="run_rw"):
            mask = detect_rolling_window(df, dc, val, window)
            _show_results(df, mask, val, display_cols)


def _show_results(df, mask, primary_col, display_cols):
    """Display anomaly results."""
    n = mask.sum()
    st.info(f"Detected **{n:,}** anomalies out of {len(df):,} rows.")

    if n == 0:
        return

    # Scatter plot
    df_plot = df.copy()
    df_plot["Anomaly"] = mask.map({True: "Anomaly", False: "Normal"})
    fig = px.scatter(df_plot, x=df_plot.index, y=primary_col, color="Anomaly",
                     color_discrete_map={"Anomaly": "#EF4444", "Normal": "#3B82F6"},
                     title=f"Anomalies in {primary_col}", opacity=0.6)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly rows
    anomalies = df[mask][display_cols]
    st.subheader("Anomaly Records")
    st.dataframe(anomalies, use_container_width=True)

    # Context rows
    with st.expander("Compare with similar normal rows"):
        context = get_context_rows(df, mask, primary_col)
        if not context.empty:
            st.dataframe(context[display_cols] if all(c in context.columns for c in display_cols)
                         else context, use_container_width=True)
        else:
            st.info("No context rows available.")

    # Export
    csv_bytes = export_anomalies_csv(anomalies)
    st.download_button("📥 Export Anomalies as CSV", csv_bytes,
                       "anomalies.csv", "text/csv", key="dl_anom")
