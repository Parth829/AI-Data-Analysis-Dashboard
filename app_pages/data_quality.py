"""
Data Quality page — score gauge, dimension breakdown, column issues, cleaning log.
"""
import streamlit as st
import plotly.graph_objects as go
from core.data_quality import QualityReport
from core.data_cleaner import CleaningAction, OutlierInfo, treat_outliers, get_display_columns
from visualizations.chart_themes import quality_color, COLORS


def render_data_quality(df, schema, quality_report, cleaning_log, outlier_report):
    """Render the data quality page."""
    st.header("Data Quality")

    # ── Quality Score Gauge ────────────────────────────────────
    score = quality_report.overall_score
    color = quality_color(score)

    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Overall Quality"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 70], "color": "#FEF3C7"},
                    {"range": [70, 90], "color": "#D1FAE5"},
                    {"range": [90, 100], "color": "#A7F3D0"},
                ],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Dimension Breakdown")
        dims = [
            ("Completeness (40%)", quality_report.completeness_score),
            ("Uniqueness (20%)", quality_report.uniqueness_score),
            ("Validity (25%)", quality_report.validity_score),
            ("Consistency (15%)", quality_report.consistency_score),
        ]
        for name, val in dims:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(min(val / 100, 1.0), text=name)
            with col_b:
                st.write(f"**{val:.1f}**")

    st.markdown("---")

    # ── Column Issues ──────────────────────────────────────────
    if quality_report.issues:
        st.subheader(f"Issues Detected ({len(quality_report.issues)})")
        for issue in quality_report.issues:
            icon = "🔴" if issue.severity == "critical" else "🟡" if issue.severity == "warning" else "🔵"
            with st.expander(f"{icon} {issue.column} — {issue.issue_type}"):
                st.write(issue.description)
                st.info(f"💡 **Recommendation**: {issue.recommendation}")
    else:
        st.success("No quality issues detected!")

    st.markdown("---")

    # ── What Was Cleaned ───────────────────────────────────────
    st.subheader("What Was Cleaned")
    if cleaning_log:
        for action in cleaning_log:
            st.write(f"• {action.description}")
    else:
        st.info("No cleaning actions were needed.")

    # ── Outlier Report ─────────────────────────────────────────
    if outlier_report:
        st.markdown("---")
        st.subheader("Outlier Report")
        st.caption("Outliers are flagged but NOT modified. Use the button below to treat them.")
        for o in outlier_report:
            st.write(f"• **{o.column}**: {o.count:,} outliers ({o.pct:.1f}%), bounds: [{o.lower_bound:.2g}, {o.upper_bound:.2g}]")

        if st.button("Treat Outliers (IQR Capping)", key="treat_outliers"):
            df_treated, report = treat_outliers(df)
            if report:
                for r in report:
                    st.success(r)
                st.session_state["df_clean"] = df_treated
                st.rerun()
            else:
                st.info("No outliers to treat.")

    # ── Reset Button ───────────────────────────────────────────
    st.markdown("---")
    if st.button("Reset to Raw Data", key="reset_data"):
        if "df_raw" in st.session_state:
            st.session_state.pop("df_clean", None)
            st.session_state.pop("schema", None)
            st.session_state.pop("quality_report", None)
            st.rerun()

    # ── Schema Overview ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Column Schema")
    import pandas as pd
    schema_rows = []
    for col, prof in schema.items():
        if col.startswith("_is_outlier_"):
            continue
        schema_rows.append({
            "Column": col,
            "Type": prof.dtype,
            "Role": prof.role.value,
            "Unique": prof.n_unique,
            "Missing %": f"{prof.pct_missing:.1f}%",
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)
