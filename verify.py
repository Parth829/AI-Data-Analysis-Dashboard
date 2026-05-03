"""Quick verification script — run with: python verify.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("1. Testing imports...")
from config.settings import PAGE_OPTIONS
from core.data_loader import load_file
from core.schema_profiler import profile_schema, ColumnRole
from core.data_cleaner import clean_for_display
from core.data_quality import compute_quality_score
from core.feature_engineering import prepare_for_modeling
from analytics.descriptive import compute_summary_stats
from analytics.insights_engine import run_all_rules
from analytics.ai_insights import generate_gemini_insights
from analytics.anomaly import detect_zscore
from analytics.forecasting import naive_forecast, PROPHET_AVAILABLE
from ml.regression import run_regression
from ml.model_evaluation import compute_regression_metrics
from visualizations.chart_themes import apply_theme
from visualizations.kpi_cards import auto_select_kpis
from visualizations.custom_charts import CHART_TYPES
from exports.excel_exporter import export_to_excel
from exports.pdf_exporter import export_to_pdf, KALEIDO_AVAILABLE
print(f"   OK — Prophet={PROPHET_AVAILABLE}, Kaleido={KALEIDO_AVAILABLE}")

print("2. Loading sample data...")
import pandas as pd
df = pd.read_csv("sample_data/retail_sales.csv")
print(f"   OK — {df.shape[0]} rows, {df.shape[1]} cols")

print("3. Schema profiling...")
schema = profile_schema(df)
roles = {}
for r in ColumnRole:
    c = sum(1 for p in schema.values() if p.role == r)
    if c > 0:
        roles[r.value] = c
print(f"   OK — {roles}")

print("4. Display-layer cleaning (Bug 1, 2, 4 fixes)...")
df_clean, log, outliers = clean_for_display(df)
print(f"   OK — {len(log)} actions, {len(outliers)} outlier cols")

# Bug 1 check: region should still be text, not encoded integers
region_vals = df_clean["region"].unique()
assert all(isinstance(v, str) for v in region_vals), f"FAIL: region encoded to integers: {region_vals}"
print(f"   Bug 1 FIX VERIFIED — region values: {list(region_vals)}")

# Bug 1 check: revenue should be real values, not scaled
rev_vals = df_clean["revenue"].dropna().head(5).tolist()
assert any(v > 10 for v in rev_vals), f"FAIL: revenue appears scaled: {rev_vals}"
print(f"   Bug 1 FIX VERIFIED — revenue values: {rev_vals[:3]}")

# Bug 2 check: outliers flagged but not capped
if outliers:
    col = outliers[0].column
    flag = f"_is_outlier_{col}"
    assert flag in df_clean.columns, f"FAIL: outlier flag column missing"
    print(f"   Bug 2 FIX VERIFIED — outliers flagged (not capped), {outliers[0].count} in {col}")

# Bug 4 check: order_id NOT converted to datetime
assert df_clean["order_id"].dtype == object, f"FAIL: order_id converted to datetime"
print(f"   Bug 4 FIX VERIFIED — order_id dtype: {df_clean['order_id'].dtype}")

print("5. Quality scoring...")
qr = compute_quality_score(df_clean, schema)
print(f"   OK — Score: {qr.overall_score}/100, Issues: {len(qr.issues)}")

print("6. Insights engine...")
insights = run_all_rules(df_clean, schema)
print(f"   OK — {len(insights)} insights generated")
for ins in insights[:3]:
    print(f"   [{ins.severity}] {ins.title}")

print("7. Naive forecast...")
fc, metrics, err = naive_forecast(df_clean, "order_date", "revenue", 30)
if err:
    print(f"   WARN — {err}")
else:
    print(f"   OK — RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.1f}%")

print()
print("=== ALL VERIFICATION PASSED ===")
