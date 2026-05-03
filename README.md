# 📊 AI-Powered Data Analysis Dashboard

An intelligent Streamlit dashboard that transforms raw CSV/Excel/Parquet/JSON files into actionable insights, visualizations, forecasts, and AI-powered recommendations via Google Gemini.

## ✨ Features

| Page | What It Does |
|------|-------------|
| **Overview** | Adaptive KPI cards, time trends, category breakdowns, distributions, data health — all in original units |
| **Data Quality** | 0-100 quality score (completeness, uniqueness, validity, consistency), per-column issues, cleaning log |
| **Visualizations** | Auto-generated charts + custom builder with 11 chart types |
| **Correlations** | Heatmap (IDs excluded), top pairs table, interactive scatter |
| **Insights** | Rule-based diagnostics (period changes, correlations, segments) + Gemini AI insights with domain/goal targeting |
| **Anomaly Detection** | Z-Score, IQR, Isolation Forest, rolling window — with context rows and CSV export |
| **Forecasting** | Naive baseline (always available) + optional Prophet with holdout validation and accuracy comparison |
| **Regression** | Random Forest with R², MAE, MAPE, RMSE, residuals, 5-fold CV, categorical support |
| **Export** | Excel (5 sheets), PDF (fpdf2 + optional chart images), HTML (interactive) |

## 🚀 Quickstart

```bash
# 1. Clone and enter the project
cd AI-Powered-Data-Analysis-Dashboard

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

### Configuring AI Insights (optional)

AI-generated insights require a Google Gemini API key. Set it once via environment:

```bash
# In .env (create from .env.example)
GOOGLE_API_KEY=your_key_here
```

Or via Streamlit secrets for cloud deployment (`.streamlit/secrets.toml`):
```toml
GOOGLE_API_KEY = "your_key_here"
```

If no key is configured, the dashboard runs normally with rule-based statistical insights. The AI Insights button is automatically disabled.

### Optional Dependencies

```bash
# Advanced forecasting (Prophet)
pip install prophet==1.1.6

# Chart images in PDF exports
pip install kaleido==0.2.1
```

## 📁 Project Structure

```
├── app.py                      # Entry point — sidebar, routing
├── app_pages/                  # UI pages
│   ├── overview.py             # Adaptive KPI dashboard
│   ├── data_quality.py         # Quality scoring & profiling
│   ├── visualizations.py       # Auto + custom charts
│   ├── correlations.py         # Heatmap & pairs
│   ├── insights.py             # Rule-based + AI insights
│   ├── anomaly_detection.py    # 4 detection methods
│   ├── forecasting.py          # Naive + Prophet
│   ├── regression.py           # Full ML evaluation
│   └── export.py               # Excel, PDF, HTML
├── core/                       # Data processing layer
│   ├── data_loader.py          # Multi-format loader with caching
│   ├── schema_profiler.py      # Semantic column detection (11 roles)
│   ├── data_cleaner.py         # Display-safe cleaning (no scaling!)
│   ├── data_quality.py         # Quality scoring engine
│   └── feature_engineering.py  # ML-only transforms
├── analytics/                  # Analysis engines
│   ├── descriptive.py          # Smart summary stats
│   ├── insights_engine.py      # 6 rule-based insight generators
│   ├── ai_insights.py          # Gemini integration with structured output
│   ├── anomaly.py              # Detection algorithms
│   └── forecasting.py          # Naive + optional Prophet
├── ml/                         # Machine learning
│   ├── regression.py           # Random Forest with full evaluation
│   └── model_evaluation.py     # Shared metrics & plots
├── visualizations/             # Chart components
│   ├── chart_themes.py         # Consistent Plotly styling
│   ├── kpi_cards.py            # Hero metric cards with sparklines
│   ├── auto_charts.py          # Time trends, breakdowns, distributions
│   └── custom_charts.py        # 11-type chart builder
├── exports/                    # Report generators
│   ├── excel_exporter.py       # Multi-sheet Excel
│   └── pdf_exporter.py         # fpdf2 with Unicode + optional charts
├── config/
│   └── settings.py             # All env vars, constants, thresholds
├── sample_data/
│   ├── retail_sales.csv        # 10k rows, 18 months, daily
│   └── customer_churn.csv      # 5k rows, churn patterns
├── requirements.txt
└── .env.example
```

## 🔑 Key Design Decisions

### Display Layer vs Modeling Layer
The **#1 fix** in this restructure. The old code applied `LabelEncoder` + `StandardScaler` to all data before display — turning "North" into `0` and `$1,200` into `-0.34` everywhere. Now:

- **`core/data_cleaner.py`** handles display cleaning: duplicates, missing values, date coercion. No encoding, no scaling, no capping.
- **`core/feature_engineering.py`** applies transforms only when called by ML functions (regression, anomaly detection), and only on copies.

### Outlier Treatment is Opt-In
Outliers are **detected and flagged** (via `_is_outlier_` columns) but never modified automatically. Users must click "Treat Outliers" in the Data Quality tab.

### Schema Profiling with Priority Order
Columns are assigned semantic roles (DATE, MONETARY, IDENTIFIER, etc.) using a strict priority order. This drives formatting ($1,234 vs 1234.00), KPI selection, which columns to exclude from correlations, and more.

## 🔧 Migration from Old Structure

| Old File | New Location | What Changed |
|----------|-------------|-------------|
| `utils/data_cleaning.py` | `core/data_cleaner.py` | Removed LabelEncoder, StandardScaler, IQR capping |
| `utils/insights.py` | `analytics/insights_engine.py` + `analytics/ai_insights.py` | Split into rule-based + AI; structured prompts |
| `utils/visualization.py` | `visualizations/auto_charts.py` + `visualizations/custom_charts.py` | Added theming, captions, range selectors |
| `utils/export.py` | `exports/excel_exporter.py` + `exports/pdf_exporter.py` | fpdf2, Unicode, multi-sheet, correct Gemini label |
| `components/dashboard.py` | `app_pages/overview.py` | Complete rewrite — adaptive KPI dashboard |
| `components/upload.py` | `core/data_loader.py` | Added Parquet/JSON, chardet, MD5 caching, sampling |
| `models/anomaly_detection.py` | `analytics/anomaly.py` | Added rolling window, context rows, CSV export |
| `models/predictive_analytics.py` | `analytics/forecasting.py` + `ml/regression.py` | Added naive baseline, holdout, R²/MAE/MAPE, CV |

## 🔌 How to Extend

### Adding a New Insight Rule
1. Open `analytics/insights_engine.py`
2. Write a function: `def _rule_my_rule(df, schema) -> list[Insight] | None`
3. Add it to `_ALL_RULES` list

### Adding a New Chart Type
1. Open `visualizations/custom_charts.py`
2. Add the type name to `CHART_TYPES`
3. Add the case in `_create_chart()`

### Swapping the LLM
1. Open `analytics/ai_insights.py`
2. Modify `generate_gemini_insights()` to use your preferred LLM client
3. Keep the same return format: `list[dict]` with `{insight, evidence, confidence, recommendation}`

## 📋 Requirements

- Python 3.10+
- See `requirements.txt` for pinned versions
- Google API key for AI insights (optional)
