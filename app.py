"""
AI-Powered Data Analysis Dashboard — Main Entry Point.
Thin router: sidebar navigation, file loading, page routing.
"""
import hashlib
import streamlit as st
import pandas as pd

from config.settings import PAGE_OPTIONS, SUPPORTED_EXTENSIONS
from core.data_loader import load_file, load_sample_data
from core.data_cleaner import clean_for_display, get_display_columns
from core.schema_profiler import profile_schema, get_category_columns
from core.data_quality import compute_quality_score

# Page renderers
from app_pages.overview import render_overview
from app_pages.data_quality import render_data_quality
from app_pages.visualizations import render_visualizations
from app_pages.correlations import render_correlations
from app_pages.insights import render_insights
from app_pages.anomaly_detection import render_anomaly_detection
from app_pages.forecasting import render_forecasting
from app_pages.regression import render_regression
from app_pages.export import render_export

from visualizations.styles import (
    inject_css, sidebar_section, render_app_header, render_styled_error,
    SVG_BAR_CHART, SVG_SPARKLES, SVG_TRENDING_UP
)

st.set_page_config(
    page_title="AI Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dispatch dictionary
PAGE_RENDERERS = {
    "Overview": lambda df, sch, qual, log, out: render_overview(df, sch, qual),
    "Data Quality": lambda df, sch, qual, log, out: render_data_quality(df, sch, qual, log, out),
    "Visualizations": lambda df, sch, qual, log, out: render_visualizations(df, sch),
    "Correlations": lambda df, sch, qual, log, out: render_correlations(df, sch),
    "Insights": lambda df, sch, qual, log, out: render_insights(df, sch),
    "Anomaly Detection": lambda df, sch, qual, log, out: render_anomaly_detection(df, sch),
    "Forecasting": lambda df, sch, qual, log, out: render_forecasting(df, sch),
    "Regression": lambda df, sch, qual, log, out: render_regression(df, sch),
    "Export": lambda df, sch, qual, log, out: render_export(df, st.session_state.get("df_raw"), sch, log),
}

# Material Symbols
NAV_ITEMS = [
    ("Overview", ":material/dashboard:"),
    ("Data Quality", ":material/verified:"),
    ("Visualizations", ":material/insights:"),
    ("Correlations", ":material/scatter_plot:"),
    ("Insights", ":material/lightbulb:"),
    ("Anomaly Detection", ":material/warning:"),
    ("Forecasting", ":material/trending_up:"),
    ("Regression", ":material/timeline:"),
    ("Export", ":material/download:"),
]


def format_nav_item(page_name):
    """Format the radio option with a Material Symbol."""
    for name, icon in NAV_ITEMS:
        if name == page_name:
            return f"{icon} {name}"
    return page_name


def main():
    inject_css()

    # Determine filename for header
    filename = None
    
    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        # App logo / name at top
        st.markdown("<div style='font-weight:700; font-size:14px; margin-bottom: 2rem;'>✦ DATA APP</div>", unsafe_allow_html=True)

        sidebar_section("DATA")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=SUPPORTED_EXTENSIONS,
            help="Supports CSV, XLSX, XLS, Parquet, JSON",
            label_visibility="collapsed"
        )

        # Sample data option
        use_sample = st.selectbox(
            "Or try sample data:",
            ["None", "Retail Sales (10k rows)", "Customer Churn (5k rows)"],
            key="sample_choice"
        )
        
        # Resolve filename
        if uploaded_file is not None:
            filename = uploaded_file.name
        elif use_sample != "None":
            filename = "Sample: " + use_sample

    render_app_header(filename)

    # ── Load data ──────────────────────────────────────────────
    df_raw = None

    if uploaded_file is not None:
        df_raw = load_file(uploaded_file)
    elif use_sample != "None":
        try:
            fname = "retail_sales.csv" if "Retail" in use_sample else "customer_churn.csv"
            df_raw = load_sample_data(fname)
        except Exception as e:
            render_styled_error(e)

    if df_raw is None:
        _render_landing()
        return

    # Store raw for export/reset
    st.session_state["df_raw"] = df_raw

    # ── Clean & profile ────────────────────────────────────────
    if "df_clean" not in st.session_state or st.session_state.get("_data_hash") != _hash_df(df_raw):
        with st.spinner("Analyzing dataset..."):
            df_clean, cleaning_log, outlier_report = clean_for_display(df_raw)
            schema = profile_schema(df_clean)
            quality_report = compute_quality_score(df_clean, schema)

            st.session_state["df_clean"] = df_clean
            st.session_state["cleaning_log"] = cleaning_log
            st.session_state["outlier_report"] = outlier_report
            st.session_state["schema"] = schema
            st.session_state["quality_report"] = quality_report
            st.session_state["_data_hash"] = _hash_df(df_raw)

    df_clean = st.session_state["df_clean"]
    cleaning_log = st.session_state["cleaning_log"]
    outlier_report = st.session_state["outlier_report"]
    schema = st.session_state["schema"]
    quality_report = st.session_state["quality_report"]

    # ── Sidebar filters ────────────────────────────────────────
    df_filtered = df_clean.copy()
    with st.sidebar:
        st.markdown("<div class='dash-spacer-sm'></div>", unsafe_allow_html=True)
        sidebar_section("FILTERS")
        
        cat_cols = get_category_columns(schema)
        for col in cat_cols[:3]:  # limit to 3 filter columns
            if col not in df_clean.columns:
                continue
            unique_vals = df_clean[col].dropna().unique().tolist()
            if len(unique_vals) > 50:
                continue
            selected = st.multiselect(
                f"{col.replace('_', ' ').title()}",
                unique_vals, default=unique_vals,
                key=f"filter_{col}",
            )
            if selected and len(selected) < len(unique_vals):
                df_filtered = df_filtered[df_filtered[col].isin(selected)]

        if len(df_filtered) < len(df_clean):
            st.markdown(f"<div class='inline-status-pill'>{len(df_filtered):,} of {len(df_clean):,} rows</div>", unsafe_allow_html=True)

    # ── Page navigation ────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='dash-spacer-sm'></div>", unsafe_allow_html=True)
        sidebar_section("PAGES")
        page = st.radio("Navigate:", PAGE_OPTIONS, format_func=format_nav_item, key="nav_page", label_visibility="collapsed")
        
        st.markdown("<div style='margin-top: auto; padding-top: 2rem; font-size: 12px; color: var(--text-muted);'>v1.0 &middot; docs</div>", unsafe_allow_html=True)

    # ── Route to page ──────────────────────────────────────────
    try:
        renderer = PAGE_RENDERERS.get(page)
        if renderer:
            renderer(df_filtered, schema, quality_report, cleaning_log, outlier_report)
        else:
            render_styled_error(Exception(f"Unknown page: {page}"))
    except Exception as e:
        render_styled_error(e)


def _hash_df(df):
    """Quick hash of dataframe for cache invalidation."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


def _render_landing():
    """Show a landing page when no data is loaded."""
    
    # Hero
    st.markdown("<div class='dash-spacer-md'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; max-width: 600px; margin: 0 auto;">
            <h2 class="landing-hero-title">Get started by uploading data</h2>
            <p class="landing-hero-desc">Drag a CSV, Excel, Parquet, or JSON file into the sidebar &mdash; or pick a sample dataset to explore.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='dash-spacer-lg'></div>", unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{SVG_BAR_CHART}</div>
            <div class="feature-title">Adaptive Dashboard</div>
            <div class="feature-desc">KPI cards, trends, and breakdowns that respond dynamically to your schema.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{SVG_SPARKLES}</div>
            <div class="feature-title">AI Insights</div>
            <div class="feature-desc">Gemini-powered analysis with domain-specific prompting and automated summaries.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{SVG_TRENDING_UP}</div>
            <div class="feature-title">Forecasting</div>
            <div class="feature-desc">Naive baselines + Prophet forecasting with automated accuracy metrics.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='dash-spacer-lg'></div>", unsafe_allow_html=True)
    
    # Supported formats
    st.markdown(
        """
        <div style="text-align: center;">
            <span style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-right: 12px;">Supported formats</span>
            <span class="format-pill">CSV</span>
            <span class="format-pill">XLSX</span>
            <span class="format-pill">Parquet</span>
            <span class="format-pill">JSON</span>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
