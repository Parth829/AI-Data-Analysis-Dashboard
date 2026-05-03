"""
Visualizations page — auto charts + custom builder.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from core.schema_profiler import get_date_columns, get_numeric_columns, get_category_columns
from core.data_cleaner import get_display_columns
from visualizations.auto_charts import render_time_trend
from visualizations.custom_charts import render_custom_chart_builder
from visualizations.chart_themes import apply_theme


def render_visualizations(df, schema):
    """Render the visualizations page."""
    st.header("Visualizations")
    display_cols = get_display_columns(df)
    df_d = df[display_cols]

    tab_auto, tab_custom = st.tabs(["Auto-Generated", "Custom Builder"])

    with tab_auto:
        # Numeric distributions
        num_cols = df_d.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.subheader("Numeric Distributions")
            sel = st.selectbox("Select column:", num_cols, key="viz_num_col")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df_d, x=sel, title=f"Histogram — {sel}", marginal="violin")
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(df_d, y=sel, title=f"Box Plot — {sel}")
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Categorical distributions
        cat_cols = [c for c in df_d.select_dtypes(exclude=[np.number, "datetime"]).columns if df_d[c].nunique() < 50]
        if cat_cols:
            st.subheader("Categorical Distributions")
            sel = st.selectbox("Select column:", cat_cols, key="viz_cat_col")
            vc = df_d[sel].value_counts().head(20).reset_index()
            vc.columns = [sel, "Count"]
            fig = px.bar(vc, x=sel, y="Count", title=f"Top 20 — {sel}", color=sel)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Time series
        date_cols = get_date_columns(schema)
        if date_cols and pd.api.types.is_datetime64_any_dtype(df_d.get(date_cols[0])):
            st.subheader("Time Series")
            dc = date_cols[0]
            val_col = st.selectbox("Value column:", num_cols, key="viz_ts_val")
            temp = df_d.sort_values(dc)
            fig = px.line(temp, x=dc, y=val_col, title=f"{val_col} over time")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab_custom:
        render_custom_chart_builder(df_d)
