"""
Export page — Excel, PDF, HTML reports.
"""
import streamlit as st
from core.data_cleaner import get_display_columns
from analytics.insights_engine import run_all_rules
from exports.excel_exporter import export_to_excel
from exports.pdf_exporter import export_to_pdf, KALEIDO_AVAILABLE


def render_export(df, df_raw, schema, cleaning_log):
    """Render the export page."""
    st.header("Export Reports")

    # Gather insights for reports
    insights = run_all_rules(df, schema)
    cleaning_text = "\n".join(a.description for a in cleaning_log) if cleaning_log else "No cleaning needed."
    ai_insights = st.session_state.get("ai_cache", {})
    # Flatten AI insights from cache
    flat_ai = []
    for v in ai_insights.values():
        if isinstance(v, list):
            flat_ai.extend(v)

    st.subheader("Excel Report")
    st.caption("Includes: Raw Data, Cleaned Data, Summary Stats, Insights, Anomalies")
    try:
        excel_bytes = export_to_excel(df_raw, df, insights)
        st.download_button("Download Excel Report", excel_bytes,
                           "analysis_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_excel")
    except Exception as e:
        st.error(f"Excel export failed: {e}")

    st.markdown("---")

    st.subheader("PDF Report")
    if not KALEIDO_AVAILABLE:
        st.info("Install `kaleido` to include chart images in the PDF.")
    st.caption("Includes: Cleaning summary, Statistical insights, AI insights" +
               (", Charts" if KALEIDO_AVAILABLE else " (text-only)"))
    try:
        if st.button("Generate PDF", key="gen_pdf"):
            with st.spinner("Generating PDF..."):
                from visualizations.auto_charts import render_time_trend, render_category_breakdown, render_distribution
                
                export_figs = []
                if KALEIDO_AVAILABLE:
                    f1 = render_time_trend(df, schema, return_only=True)
                    if f1: export_figs.extend(f1)
                    f2 = render_category_breakdown(df, schema, return_only=True)
                    if f2: export_figs.extend(f2)
                    f3 = render_distribution(df, schema, return_only=True)
                    if f3: export_figs.extend(f3)
                    
                pdf_bytes = export_to_pdf(cleaning_text, insights, flat_ai, figures=export_figs if export_figs else None)
                st.download_button("Download PDF Report", pdf_bytes,
                                   "analysis_report.pdf", "application/pdf",
                                   key="dl_pdf")
    except Exception as e:
        st.error(f"PDF export failed: {e}")

    st.markdown("---")

    st.subheader("HTML Report")
    st.caption("Interactive report with embedded Plotly charts. Shareable as a single file.")
    if st.button("Generate HTML Report", key="gen_html"):
        try:
            html = _build_html_report(df, schema, insights, flat_ai, cleaning_text)
            st.download_button("Download HTML Report", html,
                               "analysis_report.html", "text/html",
                               key="dl_html")
        except Exception as e:
            st.error(f"HTML export failed: {e}")


def _build_html_report(df, schema, insights, ai_insights, cleaning_text):
    """Build an interactive HTML report with embedded Plotly charts."""
    from visualizations.auto_charts import render_time_trend, render_category_breakdown, render_distribution
    
    display_cols = get_display_columns(df)
    df_d = df[display_cols]

    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Data Analysis Report</title>",
        # Plotly.js CDN for interactive charts
        "<script src='https://cdn.plot.ly/plotly-2.35.0.min.js'></script>",
        "<style>",
        "body{font-family:Inter,system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:20px;background:#FAFAFA;color:#1F2937}",
        "h1{color:#4F46E5;border-bottom:3px solid #4F46E5;padding-bottom:12px}",
        "h2{color:#374151;border-bottom:2px solid #E5E7EB;padding-bottom:8px;margin-top:32px}",
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #E5E7EB;padding:8px 12px;text-align:left}",
        "th{background:#4F46E5;color:white;font-weight:600}",
        "tr:nth-child(even){background:#F9FAFB}",
        ".card{border:1px solid #E5E7EB;padding:16px;border-radius:8px;margin:8px 0;background:white;box-shadow:0 1px 3px rgba(0,0,0,0.06)}",
        ".card strong{color:#4F46E5}",
        ".chart-container{margin:16px 0;border:1px solid #E5E7EB;border-radius:8px;background:white;padding:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}",
        ".meta{color:#6B7280;font-size:0.9em}",
        "</style>",
        "</head><body>",
        "<h1>Data Analysis Report</h1>",
        f"<p class='meta'>{len(df_d)} rows × {len(display_cols)} columns</p>",
        
        # Cleaning Summary
        "<h2>Cleaning Summary</h2>",
        f"<div class='card'>{cleaning_text.replace(chr(10), '<br>')}</div>",
        
        # Statistical Insights
        "<h2>Statistical Insights</h2>",
    ]

    for ins in insights:
        sev_icon = "🔴" if ins.severity == "critical" else "🟡" if ins.severity == "warning" else "🔵"
        parts.append(f"<div class='card'>{sev_icon} <strong>{ins.title}</strong><br>{ins.description}<br>"
                     f"<em style='color:#6B7280'>💡 {ins.recommendation}</em></div>")

    # AI Insights
    parts.append("<h2>AI Insights (Gemini)</h2>")
    if ai_insights:
        for item in ai_insights:
            if isinstance(item, dict):
                conf = item.get("confidence", "medium")
                conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                parts.append(f"<div class='card'><strong>{item.get('insight','')}</strong> {conf_icon}<br>"
                             f"<span style='color:#6B7280'>{item.get('evidence','')}</span><br>"
                             f"<em style='color:#4F46E5'>→ {item.get('recommendation','')}</em></div>")
    else:
        parts.append("<div class='card'>No AI insights generated.</div>")

    # Interactive Charts
    parts.append("<h2>Interactive Charts</h2>")
    chart_figs = []
    f1 = render_time_trend(df, schema, return_only=True)
    if f1: chart_figs.extend(f1)
    f2 = render_category_breakdown(df, schema, return_only=True)
    if f2: chart_figs.extend(f2)
    f3 = render_distribution(df, schema, return_only=True)
    if f3: chart_figs.extend(f3)

    if chart_figs:
        for i, fig in enumerate(chart_figs[:6]):
            fig.update_layout(margin=dict(l=120, r=30, t=50, b=50))
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            parts.append(f"<div class='chart-container'>{chart_html}</div>")
    else:
        parts.append("<div class='card'>No charts available for this dataset.</div>")

    # Data Preview
    parts.append("<h2>Data Preview (First 20 Rows)</h2>")
    parts.append(df_d.head(20).to_html(index=False, classes="data-table"))
    parts.append("</body></html>")

    return "\n".join(parts).encode("utf-8")
