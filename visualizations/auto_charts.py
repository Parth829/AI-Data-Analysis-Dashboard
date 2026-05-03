"""
Auto-generated charts: time trends, category breakdowns, distributions, correlations.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core.schema_profiler import (
    ColumnProfile, ColumnRole, get_date_columns, get_monetary_columns,
    get_numeric_columns, get_category_columns, get_id_columns,
)
from visualizations.chart_themes import apply_theme


def render_time_trend(df, schema, return_only=False):
    """Multi-line time trend with range selector buttons."""
    date_cols = get_date_columns(schema)
    if not date_cols:
        return [] if return_only else None
    dc = date_cols[0]
    if not pd.api.types.is_datetime64_any_dtype(df[dc]):
        return [] if return_only else None

    # Pick top 1-3 numeric columns (monetary first)
    num_cols = get_monetary_columns(schema)
    if len(num_cols) < 3:
        others = [c for c in get_numeric_columns(schema) if c not in num_cols]
        num_cols = (num_cols + others)[:3]
    if not num_cols:
        return [] if return_only else None

    # Aggregate by date
    agg = df.groupby(dc)[num_cols].sum().sort_index().reset_index()

    fig = go.Figure()
    for i, col in enumerate(num_cols):
        fig.add_trace(go.Scatter(
            x=agg[dc], y=agg[col], mode="lines", name=col.replace("_", " ").title(),
        ))

    fig.update_layout(
        title="Trends Over Time",
        xaxis_title="", yaxis_title="",
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=7, label="7D", step="day", stepmode="backward"),
                dict(count=30, label="30D", step="day", stepmode="backward"),
                dict(count=90, label="90D", step="day", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="All"),
            ]),
            rangeslider=dict(visible=True),
        ),
    )
    apply_theme(fig)
    
    if return_only:
        return [fig]

    # Auto-caption
    if len(agg) > 30:
        recent = agg.tail(30)
        prior = agg.iloc[-60:-30] if len(agg) >= 60 else agg.head(30)
        col0 = num_cols[0]
        r, p = recent[col0].sum(), prior[col0].sum()
        if p > 0:
            change = (r - p) / p * 100
            direction = "grew" if change > 0 else "declined"
            caption = f"📈 {col0.replace('_',' ').title()} {direction} {abs(change):.0f}% in the last 30 days vs prior 30."
            st.caption(caption)

    st.plotly_chart(fig, use_container_width=True)


def render_category_breakdown(df, schema, return_only=False):
    """Two side-by-side horizontal bar charts of top categories."""
    cat_cols = get_category_columns(schema)
    if not cat_cols:
        return [] if return_only else None

    # Pick metric to show
    mon = get_monetary_columns(schema)
    num = get_numeric_columns(schema)
    metric = mon[0] if mon else (num[0] if num else None)

    figs = []
    if not return_only:
        cols = st.columns(min(len(cat_cols), 2))
        
    for i, cat in enumerate(cat_cols[:2]):
        if metric and metric in df.columns:
            agg = df.groupby(cat)[metric].sum().nlargest(10).reset_index()
            fig = px.bar(agg, y=cat, x=metric, orientation="h",
                         title=f"{metric.replace('_',' ').title()} by {cat.replace('_',' ').title()}")
            total = df[metric].sum()
            top3_sum = agg[metric].head(3).sum()
            caption = f"Top 3 {cat} account for {top3_sum/total*100:.0f}% of {metric}." if total > 0 else None
        else:
            vc = df[cat].value_counts().head(10).reset_index()
            vc.columns = [cat, "Count"]
            fig = px.bar(vc, y=cat, x="Count", orientation="h",
                         title=f"Top 10 — {cat.replace('_',' ').title()}")
            caption = None

        apply_theme(fig)
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        
        if return_only:
            figs.append(fig)
        else:
            with cols[i]:
                if caption:
                    st.caption(caption)
                st.plotly_chart(fig, use_container_width=True)
                
    if return_only:
        return figs


def render_distribution(df, schema, return_only=False):
    """Histogram + box overlay for primary numeric metric."""
    num_cols = get_monetary_columns(schema) + get_numeric_columns(schema)
    if not num_cols:
        return [] if return_only else None

    col = num_cols[0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[col], name="Distribution", marker_color="#4F46E5", opacity=0.7))
    fig.add_trace(go.Box(x=df[col], name="Box", marker_color="#7C3AED", boxmean=True))
    fig.update_layout(title=f"Distribution of {col.replace('_',' ').title()}", height=400, barmode="overlay")
    apply_theme(fig)
    
    if return_only:
        return [fig]
        
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_correlation(df, schema):
    """Scatter of strongest-correlated numeric pair with r badge."""
    num_cols = get_numeric_columns(schema)
    id_cols = get_id_columns(schema)
    cols = [c for c in num_cols if c not in id_cols and c in df.columns]
    if len(cols) < 2:
        return

    corr = df[cols].corr()
    # Find strongest pair
    best_val, best_pair = 0, (cols[0], cols[1])
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = abs(corr.iloc[i, j])
            if val > best_val:
                best_val = val
                best_pair = (cols[i], cols[j])

    c1, c2 = best_pair
    r = corr.loc[c1, c2]
    fig = px.scatter(df, x=c1, y=c2, opacity=0.5,
                     title=f"{c1} vs {c2}  (r = {r:.3f})")
    # Add trendline
    fig.add_trace(go.Scatter(
        x=df[c1].sort_values(), y=np.poly1d(np.polyfit(df[c1].dropna(), df[c2].dropna(), 1))(df[c1].sort_values()),
        mode="lines", name="Trend", line=dict(color="red", dash="dash"),
    ))
    apply_theme(fig)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
