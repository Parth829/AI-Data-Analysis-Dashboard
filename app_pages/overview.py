"""
Overview dashboard — adaptive KPI-driven page.
Rows: KPIs → Time trend → Category breakdown → Distribution → Health strip → Recent records.
"""
import streamlit as st
import pandas as pd
import numpy as np
from core.schema_profiler import (
    ColumnProfile, ColumnRole, get_date_columns, get_numeric_columns,
    get_monetary_columns, get_category_columns, format_value,
)
from core.data_cleaner import get_display_columns
from core.data_quality import compute_quality_score
from visualizations.kpi_cards import render_kpi_row
from visualizations.auto_charts import (
    render_time_trend, render_category_breakdown,
    render_distribution, render_scatter_correlation,
)
from visualizations.chart_themes import quality_color


def render_overview(df, schema, quality_report):
    """Render the adaptive overview dashboard."""
    st.header("Dashboard Overview")
    display_cols = get_display_columns(df)

    # ── Row 1: Hero KPI Cards ──────────────────────────────────
    render_kpi_row(df, schema)

    st.markdown("---")

    # ── Row 2: Time Trend Panel ────────────────────────────────
    date_cols = get_date_columns(schema)
    if date_cols and pd.api.types.is_datetime64_any_dtype(df[date_cols[0]]):
        render_time_trend(df, schema)
        st.markdown("---")

    # ── Row 3: Category Breakdown ──────────────────────────────
    cat_cols = get_category_columns(schema)
    if cat_cols:
        st.subheader("Category Breakdown")
        render_category_breakdown(df, schema)
        st.markdown("---")

    # ── Row 4: Distribution + Relationships ────────────────────
    num_cols = get_numeric_columns(schema)
    if num_cols:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Distribution")
            render_distribution(df, schema)
        with col_right:
            if len(num_cols) >= 2:
                st.subheader("Strongest Relationship")
                render_scatter_correlation(df, schema)

        st.markdown("---")

    # ── Row 5: Data Health Strip ───────────────────────────────
    st.subheader("Data Health")
    h1, h2, h3, h4, h5 = st.columns(5)
    with h1:
        color = quality_color(quality_report.overall_score)
        st.markdown(
            f"<div style='text-align:center;padding:8px;background:{color};color:white;"
            f"border-radius:8px;font-size:24px;font-weight:bold'>"
            f"{quality_report.overall_score:.0f}/100</div>",
            unsafe_allow_html=True,
        )
        st.caption("Quality Score")
    with h2:
        st.metric("Rows", f"{len(df):,}")
    with h3:
        st.metric("Columns", f"{len(display_cols):,}")
    with h4:
        if date_cols and pd.api.types.is_datetime64_any_dtype(df[date_cols[0]]):
            dates = df[date_cols[0]].dropna()
            if len(dates) > 0:
                st.metric("Date Range", f"{(dates.max() - dates.min()).days}d")
            else:
                st.metric("Date Range", "—")
        else:
            st.metric("Date Range", "N/A")
    with h5:
        n_issues = len(quality_report.issues)
        if n_issues > 0:
            st.warning(f"{n_issues} issues — see Data Quality tab")
        else:
            st.success("No issues ✓")

    st.markdown("---")

    # ── Row 6: Recent Records ──────────────────────────────────
    st.subheader("Recent Records")
    df_display = df[display_cols].copy()

    # Sort by date if available
    if date_cols and pd.api.types.is_datetime64_any_dtype(df_display.get(date_cols[0])):
        df_display = df_display.sort_values(date_cols[0], ascending=False)

    # Format for display
    formatted = df_display.head(5).copy()
    for col in formatted.columns:
        if col in schema:
            prof = schema[col]
            if prof.role == ColumnRole.MONETARY:
                formatted[col] = formatted[col].apply(lambda v: format_value(v, ColumnRole.MONETARY))
            elif prof.role == ColumnRole.PERCENTAGE:
                formatted[col] = formatted[col].apply(lambda v: format_value(v, ColumnRole.PERCENTAGE, prof.format_hint))
            elif prof.role == ColumnRole.DATE:
                formatted[col] = formatted[col].apply(lambda v: format_value(v, ColumnRole.DATE))

    # Prettify column names
    formatted.columns = [c.replace("_", " ").title() for c in formatted.columns]
    st.dataframe(formatted, use_container_width=True, hide_index=True)
