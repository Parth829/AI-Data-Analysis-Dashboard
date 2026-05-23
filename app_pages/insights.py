"""
Insights page — rule-based + AI insights.
"""
import streamlit as st
from analytics.insights_engine import run_all_rules
from analytics.ai_insights import generate_gemini_insights
from config.settings import DOMAIN_OPTIONS, GOAL_OPTIONS, get_api_key
from visualizations.chart_themes import severity_color


def render_insights(df, schema):
    """Render the insights page."""
    st.header("Insights")

    # ── Rule-based insights ────────────────────────────────────
    st.subheader("Statistical Insights")
    insights = run_all_rules(df, schema)

    if insights:
        for ins in insights:
            color = severity_color(ins.severity)
            icon = "🔴" if ins.severity == "critical" else "🟡" if ins.severity == "warning" else "🔵"
            with st.container():
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding:12px 16px;"
                    f"margin-bottom:8px;border-radius:4px;background-color:var(--secondary-background-color);color:var(--text-color);'>"
                    f"<strong>{icon} {ins.title}</strong><br>"
                    f"{ins.description}<br>"
                    f"<em style='color:#6B7280'>💡 {ins.recommendation}</em>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No significant statistical insights found for this dataset.")

    st.markdown("---")

    # ── AI Insights ────────────────────────────────────────────
    st.subheader("AI-Generated Insights")

    c1, c2 = st.columns(2)
    with c1:
        domain = st.selectbox("Industry context:", DOMAIN_OPTIONS, key="ai_domain")
    with c2:
        goal = st.selectbox("Analysis goal:", GOAL_OPTIONS, key="ai_goal")

    api_key = get_api_key()
    key_available = api_key is not None

    if st.button(
        "Generate AI Insights",
        key="gen_ai",
        disabled=not key_available,
        help=None if key_available else "AI insights unavailable — configure GOOGLE_API_KEY in your environment.",
    ):
        with st.spinner("Analyzing..."):
            results = generate_gemini_insights(df, schema, domain, goal)

        if results:
            for item in results:
                if isinstance(item, dict):
                    with st.container():
                        conf = item.get("confidence", "medium")
                        conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                        st.markdown(
                            f"<div style='border:1px solid #E5E7EB;padding:14px 16px;"
                            f"margin-bottom:10px;border-radius:8px;background-color:var(--secondary-background-color);color:var(--text-color);'>"
                            f"<strong>{item.get('insight', '')}</strong> {conf_icon}<br>"
                            f"<span style='color:#6B7280'>{item.get('evidence', '')}</span><br>"
                            f"<em style='color:#4F46E5'>→ {item.get('recommendation', '')}</em>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.success(str(item))
        else:
            st.warning("AI insights unavailable — showing rule-based insights above.")

