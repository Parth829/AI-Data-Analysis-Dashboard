"""
Correlations page — heatmap, top pairs, scatter.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from core.schema_profiler import get_numeric_columns, get_id_columns
from visualizations.chart_themes import apply_theme


def render_correlations(df, schema):
    """Render the correlations page."""
    st.header("Correlation Analysis")

    # Exclude identifiers
    num_cols = get_numeric_columns(schema)
    id_cols = get_id_columns(schema)
    cols = [c for c in num_cols if c not in id_cols and c in df.columns]

    if len(cols) < 2:
        st.warning("Need at least 2 non-ID numeric columns for correlation analysis.")
        return

    corr = df[cols].corr()

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation Matrix (IDs excluded)")
    fig.update_layout(height=max(400, len(cols) * 35))
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Top pairs table
    st.subheader("Top Correlation Pairs")
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"Column A": cols[i], "Column B": cols[j],
                          "Correlation": corr.iloc[i, j],
                          "Abs Correlation": abs(corr.iloc[i, j])})
    pairs_df = pd.DataFrame(pairs).sort_values("Abs Correlation", ascending=False).head(15)
    pairs_df = pairs_df.drop(columns=["Abs Correlation"])
    st.dataframe(pairs_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Interactive scatter
    st.subheader("Explore Pair")
    c1, c2 = st.columns(2)
    with c1:
        sel_a = st.selectbox("Column A:", cols, key="corr_a")
    with c2:
        sel_b = st.selectbox("Column B:", [c for c in cols if c != sel_a], key="corr_b")

    fig = px.scatter(df, x=sel_a, y=sel_b, opacity=0.5,
                     title=f"{sel_a} vs {sel_b} (r = {corr.loc[sel_a, sel_b]:.3f})")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
