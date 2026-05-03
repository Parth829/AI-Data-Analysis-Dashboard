"""
Data quality scoring: completeness, uniqueness, validity, consistency.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from config.settings import QUALITY_WEIGHTS
from core.schema_profiler import ColumnProfile, ColumnRole


@dataclass
class ColumnIssue:
    column: str
    issue_type: str
    severity: str  # "info", "warning", "critical"
    description: str
    recommendation: str


@dataclass
class QualityReport:
    overall_score: float
    completeness_score: float
    uniqueness_score: float
    validity_score: float
    consistency_score: float
    issues: list = field(default_factory=list)


def compute_quality_score(df: pd.DataFrame, schema: dict[str, ColumnProfile]) -> QualityReport:
    """Compute 0-100 quality score with per-dimension breakdown."""
    display_cols = [c for c in df.columns if not c.startswith("_is_outlier_")]
    df_disp = df[display_cols]
    issues = []

    # Completeness (% non-null)
    total_cells = df_disp.shape[0] * df_disp.shape[1]
    non_null = df_disp.notna().sum().sum()
    completeness = (non_null / total_cells * 100) if total_cells > 0 else 100
    for col in df_disp.columns:
        pct_miss = df_disp[col].isna().mean() * 100
        if pct_miss > 5:
            sev = "critical" if pct_miss > 30 else "warning"
            issues.append(ColumnIssue(col, "missing", sev,
                f"'{col}' has {pct_miss:.1f}% missing values.",
                f"Consider imputing or investigating why '{col}' has gaps."))

    # Uniqueness (no unexpected full-row duplicates)
    dup_count = df_disp.duplicated().sum()
    dup_pct = dup_count / len(df_disp) * 100 if len(df_disp) > 0 else 0
    uniqueness = 100 - dup_pct
    if dup_pct > 1:
        issues.append(ColumnIssue("(all)", "duplicates", "warning",
            f"{dup_count:,} duplicate rows ({dup_pct:.1f}%).",
            "Review and deduplicate if unintentional."))

    # Validity (values matching expected type/range)
    validity_scores = []
    for col, prof in schema.items():
        if col not in df_disp.columns:
            continue
        series = df_disp[col].dropna()
        if len(series) == 0:
            validity_scores.append(100)
            continue
        if prof.role == ColumnRole.DATE:
            if pd.api.types.is_datetime64_any_dtype(series):
                validity_scores.append(100)
            else:
                parsed = pd.to_datetime(series, errors="coerce")
                rate = parsed.notna().sum() / len(series) * 100
                validity_scores.append(rate)
                if rate < 90:
                    issues.append(ColumnIssue(col, "validity", "warning",
                        f"'{col}' has {100-rate:.1f}% unparseable date values.",
                        "Check for mixed formats or non-date entries."))
        elif prof.role in (ColumnRole.MONETARY, ColumnRole.QUANTITY, ColumnRole.NUMERIC_OTHER):
            if pd.api.types.is_numeric_dtype(series):
                validity_scores.append(100)
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                rate = numeric.notna().sum() / len(series) * 100
                validity_scores.append(rate)
        else:
            validity_scores.append(100)
    validity = np.mean(validity_scores) if validity_scores else 100

    # Consistency (no mixed types within column)
    consistency_scores = []
    for col in df_disp.columns:
        series = df_disp[col].dropna()
        if len(series) == 0:
            consistency_scores.append(100)
            continue
        types = series.apply(type).nunique()
        if types <= 1:
            consistency_scores.append(100)
        else:
            dominant_pct = series.apply(type).value_counts(normalize=True).iloc[0] * 100
            consistency_scores.append(dominant_pct)
            if dominant_pct < 95:
                issues.append(ColumnIssue(col, "consistency", "warning",
                    f"'{col}' has mixed types ({types} different types).",
                    f"Ensure '{col}' has a consistent data type."))
    consistency = np.mean(consistency_scores) if consistency_scores else 100

    overall = (
        completeness * QUALITY_WEIGHTS["completeness"]
        + uniqueness * QUALITY_WEIGHTS["uniqueness"]
        + validity * QUALITY_WEIGHTS["validity"]
        + consistency * QUALITY_WEIGHTS["consistency"]
    )

    return QualityReport(
        overall_score=round(overall, 1),
        completeness_score=round(completeness, 1),
        uniqueness_score=round(uniqueness, 1),
        validity_score=round(validity, 1),
        consistency_score=round(consistency, 1),
        issues=issues,
    )
