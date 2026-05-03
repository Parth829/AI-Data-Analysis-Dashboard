"""
Display-layer data cleaning.
ONLY handles: duplicates, missing values, date coercion.
NEVER encodes, scales, or caps outliers on the display dataframe.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from config.settings import DATE_PARSE_THRESHOLD, OUTLIER_IQR_MULTIPLIER


@dataclass
class CleaningAction:
    action_type: str
    description: str
    affected_rows: int = 0
    affected_cols: list = field(default_factory=list)


@dataclass
class OutlierInfo:
    column: str
    count: int
    lower_bound: float
    upper_bound: float
    pct: float


def clean_for_display(df_raw: pd.DataFrame):
    """
    Clean dataframe for display. No encoding, no scaling, no capping.
    Returns (df_clean, log: list[CleaningAction], outlier_report: list[OutlierInfo])
    """
    df = df_raw.copy()
    log = []
    outlier_report = []

    # 1. Remove duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(CleaningAction("duplicates", f"Removed {dup_count:,} duplicate rows.", dup_count))

    # 2. Impute missing — numeric with median, categorical with mode
    for col in df.select_dtypes(include=[np.number]).columns:
        n = df[col].isna().sum()
        if n > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.append(CleaningAction("impute_numeric", f"Filled {n:,} missing in '{col}' with median ({med:.2g}).", n, [col]))

    for col in df.select_dtypes(include=["object", "category"]).columns:
        n = df[col].isna().sum()
        if n > 0:
            mode = df[col].mode()
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill)
            log.append(CleaningAction("impute_categorical", f"Filled {n:,} missing in '{col}' with '{fill}'.", n, [col]))

    # 3. Date inference with 80% threshold (Bug 4 fix)
    for col in df.select_dtypes(include=["object"]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce")
        rate = parsed.notna().sum() / len(non_null)
        if rate >= DATE_PARSE_THRESHOLD:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if hasattr(df[col].dtype, "tz") and df[col].dtype.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
            log.append(CleaningAction("date_convert", f"Converted '{col}' to datetime ({rate:.0%} parsed).", int(parsed.notna().sum()), [col]))

    # 4. Outlier detection only — flag but don't modify (Bug 2 fix)
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        n_out = mask.sum()
        if n_out > 0:
            flag = f"_is_outlier_{col}"
            df[flag] = mask
            pct = n_out / len(df) * 100
            outlier_report.append(OutlierInfo(col, n_out, lo, hi, pct))
            log.append(CleaningAction("outlier_flag", f"Flagged {n_out:,} outliers in '{col}' ({pct:.1f}%).", n_out, [col, flag]))

    return df, log, outlier_report


def treat_outliers(df, columns=None):
    """Opt-in IQR capping. Returns (df, report_strings)."""
    df = df.copy()
    report = []
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith("_is_outlier_")]
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
        n = ((df[col] < lo) | (df[col] > hi)).sum()
        if n > 0:
            df.loc[df[col] < lo, col] = lo
            df.loc[df[col] > hi, col] = hi
            report.append(f"Capped {n} outliers in '{col}' to [{lo:.2g}, {hi:.2g}].")
            flag = f"_is_outlier_{col}"
            if flag in df.columns:
                df[flag] = False
    return df, report


def get_display_columns(df):
    """Return columns excluding internal outlier flags."""
    return [c for c in df.columns if not c.startswith("_is_outlier_")]
