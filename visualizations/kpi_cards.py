"""
Reusable KPI metric cards for the Overview dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.schema_profiler import (
    ColumnProfile, ColumnRole, format_value, get_monetary_columns,
    get_numeric_columns, get_date_columns,
)


def render_kpi_card(label, value_str, delta_str=None, delta_positive=None, sparkline_data=None):
    """Render a single KPI card using st.metric + optional sparkline."""
    if delta_str:
        st.metric(label=label, value=value_str, delta=delta_str,
                  delta_color="normal" if delta_positive else "inverse")
    else:
        st.metric(label=label, value=value_str)

    # Sparkline
    if sparkline_data is not None and len(sparkline_data) > 2:
        fig = go.Figure(go.Scatter(
            y=sparkline_data, mode="lines",
            line=dict(color="#4F46E5", width=2),
            fill="tozeroy", fillcolor="rgba(79,70,229,0.1)",
        ))
        fig.update_layout(
            height=50, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def auto_select_kpis(df, schema):
    """
    Auto-pick top 4-6 KPIs based on schema roles.
    Priority: MONETARY > QUANTITY > NUMERIC_OTHER.
    Returns list of dicts: {col, label, role, total, mean, delta_pct, sparkline}.
    """
    kpis = []
    date_cols = get_date_columns(schema)
    date_col = date_cols[0] if date_cols else None
    has_dates = date_col and pd.api.types.is_datetime64_any_dtype(df[date_col])

    # Collect candidates in priority order
    candidates = []
    for col, prof in schema.items():
        if prof.role == ColumnRole.MONETARY:
            candidates.append((col, prof, 0))
        elif prof.role == ColumnRole.QUANTITY:
            candidates.append((col, prof, 1))
        elif prof.role == ColumnRole.NUMERIC_OTHER:
            candidates.append((col, prof, 2))

    candidates.sort(key=lambda x: x[2])

    for col, prof, _ in candidates[:6]:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        kpi = {
            "col": col,
            "label": col.replace("_", " ").title(),
            "role": prof.role,
            "total": series.sum(),
            "mean": series.mean(),
            "format_hint": prof.format_hint,
        }

        # Period-over-period delta
        if has_dates:
            max_d = df[date_col].max()
            cutoff = max_d - pd.Timedelta(days=30)
            prior_cutoff = cutoff - pd.Timedelta(days=30)
            recent = df[(df[date_col] >= cutoff)][col].sum()
            prior = df[(df[date_col] >= prior_cutoff) & (df[date_col] < cutoff)][col].sum()
            if prior != 0:
                kpi["delta_pct"] = (recent - prior) / abs(prior) * 100
            else:
                kpi["delta_pct"] = None

            # Sparkline — daily aggregation
            daily = df.set_index(date_col)[col].resample("D").sum().dropna()
            kpi["sparkline"] = daily.values[-60:] if len(daily) > 2 else None
        else:
            kpi["delta_pct"] = None
            kpi["sparkline"] = None

        kpis.append(kpi)

    return kpis


def render_kpi_row(df, schema):
    """Render the hero KPI card row."""
    kpis = auto_select_kpis(df, schema)
    if not kpis:
        st.info("No numeric columns detected for KPI cards.")
        return

    cols = st.columns(min(len(kpis), 6))
    for i, kpi in enumerate(kpis[:6]):
        with cols[i]:
            # Format value
            if kpi["role"] == ColumnRole.MONETARY:
                val_str = format_value(kpi["total"], ColumnRole.MONETARY)
                label = f"Total {kpi['label']}"
            elif kpi["role"] == ColumnRole.QUANTITY:
                val_str = f"{int(kpi['total']):,}"
                label = f"Total {kpi['label']}"
            else:
                val_str = f"{kpi['mean']:,.2f}"
                label = f"Avg {kpi['label']}"

            # Delta
            delta_str = None
            delta_pos = None
            if kpi["delta_pct"] is not None:
                delta_str = f"{kpi['delta_pct']:+.1f}% vs prior 30d"
                delta_pos = kpi["delta_pct"] >= 0

            render_kpi_card(label, val_str, delta_str, delta_pos, kpi.get("sparkline"))
