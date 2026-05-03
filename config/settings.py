"""
Centralized configuration for the AI Data Analysis Dashboard.
All environment variables, constants, and defaults live here.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Gemini / LLM ────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# ── Data cleaning thresholds ────────────────────────────────────
DATE_PARSE_THRESHOLD = 0.80          # ≥80% non-null must parse as dates
OUTLIER_IQR_MULTIPLIER = 1.5
MAX_CATEGORY_UNIQUE_LOW = 20         # <20 unique → CATEGORY_LOW
MAX_CATEGORY_UNIQUE_HIGH = 500       # 20-500 → CATEGORY_HIGH
IDENTIFIER_UNIQUENESS_RATIO = 0.90   # >90% unique → IDENTIFIER candidate

# ── Quality scoring weights ─────────────────────────────────────
QUALITY_WEIGHTS = {
    "completeness": 0.40,
    "uniqueness": 0.20,
    "validity": 0.25,
    "consistency": 0.15,
}

# ── Forecasting ─────────────────────────────────────────────────
DEFAULT_FORECAST_PERIODS = 30
MIN_FORECAST_DATAPOINTS = 10

# ── File handling ───────────────────────────────────────────────
LARGE_FILE_THRESHOLD_MB = 100
SAMPLE_SIZES = {
    "50k": 50_000,
    "10k": 10_000,
}
SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls", "parquet", "json"]

# ── UI ──────────────────────────────────────────────────────────
PAGE_OPTIONS = [
    "Overview",
    "Data Quality",
    "Visualizations",
    "Correlations",
    "Insights",
    "Anomaly Detection",
    "Forecasting",
    "Regression",
    "Export",
]

DOMAIN_OPTIONS = [
    "General",
    "Retail / E-commerce",
    "SaaS / Software",
    "Finance / Banking",
    "Healthcare",
    "Marketing",
]

GOAL_OPTIONS = [
    "General analysis",
    "Find growth opportunities",
    "Identify risks",
    "Explain recent trends",
    "Recommend actions",
]

# ── API key helpers ─────────────────────────────────────────

def get_api_key():
    """
    Retrieve the Google API key from backend configuration only.
    Priority order:
      1. GOOGLE_API_KEY environment variable (loaded from .env via python-dotenv)
      2. Streamlit secrets (for Streamlit Cloud / secrets.toml deployments)
    Returns None if neither is set. No UI fallback.
    """
    # 1. Environment variable
    key = os.getenv("GOOGLE_API_KEY", "")
    if key and key.strip() and key.strip() != "your_api_key_here":
        return key.strip()

    # 2. Streamlit secrets
    try:
        import streamlit as st
        key = st.secrets.get("GOOGLE_API_KEY", "")
        if key and key.strip():
            return key.strip()
    except Exception:
        pass

    return None
