"""
Descriptive statistics — smart stats based on semantic roles.
"""
import numpy as np
import pandas as pd
from core.schema_profiler import (
    ColumnProfile, ColumnRole, get_date_columns, get_monetary_columns,
    get_numeric_columns, get_category_columns,
)


def compute_summary_stats(df, schema):
    """
    Compute role-aware summary statistics.
    Returns dict with keys: monetary_stats, numeric_stats, category_stats,
    date_range, period_comparison (if date column exists).
    """
    result = {}
    display_cols = [c for c in df.columns if not c.startswith("_is_outlier_")]
    df_d = df[display_cols]

    # Monetary stats
    mon_cols = get_monetary_columns(schema)
    if mon_cols:
        stats = {}
        for col in mon_cols:
            s = df_d[col].dropna()
            stats[col] = {"total": s.sum(), "mean": s.mean(), "median": s.median(),
                          "min": s.min(), "max": s.max(), "std": s.std()}
        result["monetary_stats"] = stats

    # Numeric stats
    num_cols = get_numeric_columns(schema)
    if num_cols:
        stats = {}
        for col in num_cols:
            if col in mon_cols:
                continue
            s = df_d[col].dropna()
            stats[col] = {"total": s.sum(), "mean": s.mean(), "median": s.median(),
                          "min": s.min(), "max": s.max(), "std": s.std()}
        result["numeric_stats"] = stats

    # Category stats
    cat_cols = get_category_columns(schema)
    if cat_cols:
        stats = {}
        for col in cat_cols[:5]:  # limit to 5
            vc = df_d[col].value_counts()
            stats[col] = {"n_unique": df_d[col].nunique(),
                          "top_value": vc.index[0] if len(vc) > 0 else None,
                          "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                          "top_pct": float(vc.iloc[0] / len(df_d) * 100) if len(vc) > 0 else 0}
        result["category_stats"] = stats

    # Date range
    date_cols = get_date_columns(schema)
    if date_cols:
        dc = date_cols[0]
        if pd.api.types.is_datetime64_any_dtype(df_d[dc]):
            dates = df_d[dc].dropna()
            if len(dates) > 0:
                result["date_range"] = {"column": dc, "min": dates.min(), "max": dates.max(),
                                        "span_days": (dates.max() - dates.min()).days}
                # Period comparison (last 30 days vs prior 30)
                result["period_comparison"] = _period_comparison(df_d, dc, mon_cols + [c for c in num_cols if c not in mon_cols])

    return result


def _period_comparison(df, date_col, metric_cols, days=30):
    """Compare last N days vs prior N days."""
    dates = df[date_col].dropna()
    if len(dates) == 0:
        return {}
    max_date = dates.max()
    cutoff = max_date - pd.Timedelta(days=days)
    prior_cutoff = cutoff - pd.Timedelta(days=days)

    recent = df[(df[date_col] >= cutoff) & (df[date_col] <= max_date)]
    prior = df[(df[date_col] >= prior_cutoff) & (df[date_col] < cutoff)]

    comparisons = {}
    for col in metric_cols[:6]:
        if col not in df.columns:
            continue
        r_sum = recent[col].sum()
        p_sum = prior[col].sum()
        if p_sum != 0:
            change_pct = (r_sum - p_sum) / abs(p_sum) * 100
        else:
            change_pct = 0
        comparisons[col] = {"recent": r_sum, "prior": p_sum, "change_pct": change_pct}
    return comparisons
