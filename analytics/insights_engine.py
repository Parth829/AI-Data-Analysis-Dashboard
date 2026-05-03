"""
Rule-based insights engine. Each rule is a function: (df, schema) -> Optional[Insight].
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from core.schema_profiler import (
    ColumnProfile, ColumnRole, get_date_columns, get_monetary_columns,
    get_numeric_columns, get_category_columns, get_id_columns,
)


@dataclass
class Insight:
    title: str
    severity: str  # "info", "warning", "critical"
    description: str
    recommendation: str
    affected_columns: list
    evidence: str = ""


# ── Individual rules ───────────────────────────────────────────

def _rule_period_change(df, schema):
    """Detect significant period-over-period changes."""
    insights = []
    date_cols = get_date_columns(schema)
    if not date_cols:
        return insights
    dc = date_cols[0]
    if not pd.api.types.is_datetime64_any_dtype(df[dc]):
        return insights

    dates = df[dc].dropna()
    if len(dates) < 10:
        return insights
    max_date = dates.max()
    cutoff = max_date - pd.Timedelta(days=30)
    prior_cutoff = cutoff - pd.Timedelta(days=30)
    recent = df[(df[dc] >= cutoff) & (df[dc] <= max_date)]
    prior = df[(df[dc] >= prior_cutoff) & (df[dc] < cutoff)]

    for col in get_monetary_columns(schema) + get_numeric_columns(schema):
        if col not in df.columns:
            continue
        r, p = recent[col].sum(), prior[col].sum()
        if p == 0:
            continue
        change = (r - p) / abs(p) * 100
        if abs(change) >= 15:
            direction = "increased" if change > 0 else "dropped"
            sev = "critical" if abs(change) >= 30 else "warning"
            insights.append(Insight(
                title=f"{col} {direction} {abs(change):.0f}%",
                severity=sev,
                description=f"'{col}' {direction} by {abs(change):.1f}% in the last 30 days vs prior 30 days.",
                recommendation=f"Investigate which categories or segments drove this change in '{col}'.",
                affected_columns=[col, dc],
                evidence=f"Recent: {r:,.0f}, Prior: {p:,.0f}",
            ))
    return insights


def _rule_correlation(df, schema):
    """Find strong correlations."""
    insights = []
    num_cols = get_numeric_columns(schema)
    id_cols = get_id_columns(schema)
    cols = [c for c in num_cols if c not in id_cols and c in df.columns]
    if len(cols) < 2:
        return insights

    corr = df[cols].corr()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.75:
                c1, c2 = cols[i], cols[j]
                direction = "positive" if val > 0 else "negative"
                insights.append(Insight(
                    title=f"Strong correlation: {c1} ↔ {c2}",
                    severity="info",
                    description=f"Strong {direction} correlation ({val:.2f}) between '{c1}' and '{c2}'.",
                    recommendation=f"This suggests a relationship — verify with controlled analysis before acting.",
                    affected_columns=[c1, c2],
                    evidence=f"r = {val:.3f}",
                ))
    return insights


def _rule_category_dominance(df, schema):
    """Flag categories where one value dominates."""
    insights = []
    for col in get_category_columns(schema):
        if col not in df.columns:
            continue
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > 0.60:
            insights.append(Insight(
                title=f"'{vc.index[0]}' dominates '{col}'",
                severity="info",
                description=f"In '{col}', '{vc.index[0]}' accounts for {vc.iloc[0]*100:.1f}% of all records.",
                recommendation=f"Consider whether this imbalance is expected or indicates data collection bias.",
                affected_columns=[col],
                evidence=f"{vc.iloc[0]*100:.1f}% concentration",
            ))
    return insights


def _rule_segment_opportunity(df, schema):
    """Find segments with high value but low volume."""
    insights = []
    cat_cols = get_category_columns(schema)
    mon_cols = get_monetary_columns(schema)
    if not cat_cols or not mon_cols:
        return insights

    metric = mon_cols[0]
    for cat in cat_cols[:2]:
        if cat not in df.columns or metric not in df.columns:
            continue
        grouped = df.groupby(cat)[metric].agg(["mean", "count"])
        overall_mean = df[metric].mean()
        for seg, row in grouped.iterrows():
            if row["mean"] > overall_mean * 2 and row["count"] < len(df) * 0.15:
                ratio = row["mean"] / overall_mean
                pct = row["count"] / len(df) * 100
                insights.append(Insight(
                    title=f"High-value segment: {seg}",
                    severity="info",
                    description=f"'{seg}' in '{cat}' has {ratio:.1f}× the avg {metric} but only {pct:.0f}% of records.",
                    recommendation=f"Opportunity to expand this segment for higher returns.",
                    affected_columns=[cat, metric],
                    evidence=f"Avg: {row['mean']:,.0f} vs overall {overall_mean:,.0f}",
                ))
    return insights


def _rule_skewness(df, schema):
    """Flag highly skewed distributions."""
    insights = []
    for col in get_numeric_columns(schema):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 20:
            continue
        skew = s.skew()
        if abs(skew) > 2:
            direction = "right" if skew > 0 else "left"
            insights.append(Insight(
                title=f"'{col}' is heavily skewed",
                severity="info",
                description=f"'{col}' has a strong {direction}-skew (skewness: {skew:.2f}).",
                recommendation=f"Consider log transform for modeling, or investigate extreme values.",
                affected_columns=[col],
                evidence=f"Skewness = {skew:.2f}",
            ))
    return insights


def _rule_outlier_summary(df, schema):
    """Summarize flagged outliers."""
    insights = []
    outlier_cols = [c for c in df.columns if c.startswith("_is_outlier_")]
    for flag_col in outlier_cols:
        col = flag_col.replace("_is_outlier_", "")
        n = df[flag_col].sum()
        if n > 0:
            pct = n / len(df) * 100
            sev = "warning" if pct > 5 else "info"
            insights.append(Insight(
                title=f"Outliers in '{col}'",
                severity=sev,
                description=f"{n:,} values ({pct:.1f}%) in '{col}' are flagged as outliers.",
                recommendation=f"Review outliers in the Data Quality tab. Treat only if they're errors.",
                affected_columns=[col],
                evidence=f"{n:,} outliers ({pct:.1f}%)",
            ))
    return insights


# ── Runner ─────────────────────────────────────────────────────

_ALL_RULES = [
    _rule_period_change,
    _rule_correlation,
    _rule_category_dominance,
    _rule_segment_opportunity,
    _rule_skewness,
    _rule_outlier_summary,
]


def run_all_rules(df, schema):
    """Run all insight rules and return sorted by severity."""
    results = []
    for rule in _ALL_RULES:
        try:
            out = rule(df, schema)
            if out:
                results.extend(out if isinstance(out, list) else [out])
        except Exception:
            continue

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    results.sort(key=lambda i: severity_order.get(i.severity, 3))
    return results
