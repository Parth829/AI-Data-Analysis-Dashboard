"""
Semantic schema profiling — detects column roles beyond simple dtype.
Priority order (first match wins):
  DATE > BOOLEAN > IDENTIFIER > GEOGRAPHIC > MONETARY > PERCENTAGE >
  QUANTITY > CATEGORY_LOW > CATEGORY_HIGH > FREE_TEXT > NUMERIC_OTHER
"""
import re
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from config.settings import (
    DATE_PARSE_THRESHOLD,
    IDENTIFIER_UNIQUENESS_RATIO,
    MAX_CATEGORY_UNIQUE_LOW,
    MAX_CATEGORY_UNIQUE_HIGH,
)


class ColumnRole(Enum):
    DATE = "date"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    GEOGRAPHIC = "geographic"
    MONETARY = "monetary"
    PERCENTAGE = "percentage"
    QUANTITY = "quantity"
    CATEGORY_LOW = "category_low"
    CATEGORY_HIGH = "category_high"
    FREE_TEXT = "free_text"
    NUMERIC_OTHER = "numeric_other"


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: ColumnRole
    format_hint: str = ""  # e.g. "currency", "percent", "date", "integer"
    n_unique: int = 0
    pct_missing: float = 0.0


# ── Regex patterns for name-based detection ────────────────────

_ID_PATTERN = re.compile(r"(^id$|_id$|_key$|_code$|_number$|^key$|^code$)", re.I)
_GEO_PATTERN = re.compile(
    r"(country|state|province|city|region|zip|postal|lat|lng|longitude|latitude|address)", re.I
)
_MONETARY_PATTERN = re.compile(
    r"(revenue|sales|price|cost|amount|profit|fee|salary|wage|income|budget|spend|total_price|unit_price)", re.I
)
_PCT_PATTERN = re.compile(r"(pct|percent|ratio|rate|proportion|discount)", re.I)
_QTY_PATTERN = re.compile(r"(count|qty|quantity|num_|n_|total_|units|orders|items|tickets)", re.I)


# ── Role detection functions (priority order) ──────────────────

def _is_date(series: pd.Series) -> bool:
    """Check if column is datetime or can be parsed as datetime (≥80% success)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce")
    success_rate = parsed.notna().sum() / len(non_null)
    return success_rate >= DATE_PARSE_THRESHOLD


def _is_boolean(series: pd.Series) -> bool:
    """Exactly 2 unique non-null values."""
    non_null = series.dropna()
    if non_null.nunique() != 2:
        return False
    # Check common boolean patterns
    unique_vals = set(str(v).lower().strip() for v in non_null.unique())
    bool_patterns = [
        {"true", "false"}, {"yes", "no"}, {"0", "1"},
        {"y", "n"}, {"t", "f"}, {"1.0", "0.0"},
    ]
    if unique_vals in bool_patterns:
        return True
    # Also accept any 2-value column with bool dtype
    if series.dtype == bool:
        return True
    return non_null.nunique() == 2


def _is_identifier(series: pd.Series, col_name: str) -> bool:
    """High uniqueness + id-like name or integer type."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    uniqueness = non_null.nunique() / len(non_null)
    if uniqueness <= IDENTIFIER_UNIQUENESS_RATIO:
        return False
    # Name-based check
    if _ID_PATTERN.search(col_name):
        return True
    # Integer with high uniqueness
    if pd.api.types.is_integer_dtype(series) and uniqueness > 0.95:
        return True
    return False


def _is_geographic(col_name: str) -> bool:
    return bool(_GEO_PATTERN.search(col_name))


def _is_monetary(series: pd.Series, col_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    return bool(_MONETARY_PATTERN.search(col_name))


def _is_percentage(series: pd.Series, col_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    if not _PCT_PATTERN.search(col_name):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    # Check if values are in 0–1 or 0–100 range
    mn, mx = non_null.min(), non_null.max()
    return (0 <= mn and mx <= 1) or (0 <= mn and mx <= 100)


def _is_quantity(series: pd.Series, col_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    # Integer-valued and non-negative
    is_int_like = np.allclose(non_null, non_null.round(), equal_nan=True)
    return is_int_like and non_null.min() >= 0 and _QTY_PATTERN.search(col_name) is not None


# ── Main profiler ──────────────────────────────────────────────

def _detect_role(series: pd.Series, col_name: str) -> tuple[ColumnRole, str]:
    """
    Detect semantic role for a single column.
    Returns (role, format_hint).
    Priority: DATE > BOOLEAN > IDENTIFIER > GEOGRAPHIC > MONETARY >
              PERCENTAGE > QUANTITY > CATEGORY_LOW > CATEGORY_HIGH >
              FREE_TEXT > NUMERIC_OTHER
    """
    # 1. DATE
    if _is_date(series):
        return ColumnRole.DATE, "date"

    # 2. BOOLEAN
    if _is_boolean(series):
        return ColumnRole.BOOLEAN, "boolean"

    # 3. IDENTIFIER
    if _is_identifier(series, col_name):
        return ColumnRole.IDENTIFIER, "id"

    # 4. GEOGRAPHIC
    if _is_geographic(col_name):
        if pd.api.types.is_numeric_dtype(series):
            return ColumnRole.GEOGRAPHIC, "coordinate"
        return ColumnRole.GEOGRAPHIC, "geo_name"

    # 5. MONETARY
    if _is_monetary(series, col_name):
        return ColumnRole.MONETARY, "currency"

    # 6. PERCENTAGE
    if _is_percentage(series, col_name):
        non_null = series.dropna()
        mx = non_null.max()
        hint = "pct_decimal" if mx <= 1 else "pct_whole"
        return ColumnRole.PERCENTAGE, hint

    # 7. QUANTITY
    if _is_quantity(series, col_name):
        return ColumnRole.QUANTITY, "integer"

    # 8-10. Categorical / Free text (non-numeric)
    if not pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique < MAX_CATEGORY_UNIQUE_LOW:
            return ColumnRole.CATEGORY_LOW, "category"
        if n_unique < MAX_CATEGORY_UNIQUE_HIGH:
            return ColumnRole.CATEGORY_HIGH, "category"
        # Check if free text (long strings)
        non_null = series.dropna()
        if len(non_null) > 0:
            median_len = non_null.astype(str).str.len().median()
            if median_len > 50 or n_unique > 500:
                return ColumnRole.FREE_TEXT, "text"
        return ColumnRole.CATEGORY_HIGH, "category"

    # 11. NUMERIC_OTHER — fallback
    return ColumnRole.NUMERIC_OTHER, "number"


def profile_schema(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    """
    Profile all columns in the dataframe.
    Returns dict mapping column name -> ColumnProfile.
    """
    profiles = {}
    for col in df.columns:
        role, hint = _detect_role(df[col], col)
        profiles[col] = ColumnProfile(
            name=col,
            dtype=str(df[col].dtype),
            role=role,
            format_hint=hint,
            n_unique=df[col].nunique(),
            pct_missing=df[col].isna().mean() * 100,
        )
    return profiles


# ── Formatting ─────────────────────────────────────────────────

def format_value(value, role: ColumnRole, format_hint: str = "") -> str:
    """
    Format a value for display based on its semantic role.
    """
    if pd.isna(value):
        return "—"
    try:
        if role == ColumnRole.MONETARY:
            return f"${value:,.2f}" if abs(value) < 1e9 else f"${value:,.0f}"
        if role == ColumnRole.PERCENTAGE:
            if format_hint == "pct_decimal":
                return f"{value * 100:.1f}%"
            return f"{value:.1f}%"
        if role == ColumnRole.DATE:
            if isinstance(value, (pd.Timestamp,)):
                return value.strftime("%b %d, %Y")
            return str(value)
        if role == ColumnRole.QUANTITY:
            return f"{int(value):,}"
        if role == ColumnRole.NUMERIC_OTHER:
            if isinstance(value, float):
                return f"{value:,.2f}" if abs(value) < 1e6 else f"{value:,.0f}"
            return f"{value:,}"
        return str(value)
    except (ValueError, TypeError):
        return str(value)


def get_date_columns(schema: dict[str, ColumnProfile]) -> list[str]:
    """Return list of column names with DATE role."""
    return [p.name for p in schema.values() if p.role == ColumnRole.DATE]


def get_numeric_columns(schema: dict[str, ColumnProfile]) -> list[str]:
    """Return numeric columns excluding identifiers."""
    return [
        p.name for p in schema.values()
        if p.role in (ColumnRole.MONETARY, ColumnRole.QUANTITY, ColumnRole.PERCENTAGE, ColumnRole.NUMERIC_OTHER)
    ]


def get_monetary_columns(schema: dict[str, ColumnProfile]) -> list[str]:
    """Return monetary columns, prioritizing revenue/sales/profit over unit_price/cost."""
    import re
    primary = re.compile(r"(revenue|sales|profit|income|total)", re.I)
    cols = [p.name for p in schema.values() if p.role == ColumnRole.MONETARY]
    # Sort: primary keywords first
    cols.sort(key=lambda c: (0 if primary.search(c) else 1, c))
    return cols


def get_category_columns(schema: dict[str, ColumnProfile]) -> list[str]:
    return [p.name for p in schema.values() if p.role in (ColumnRole.CATEGORY_LOW, ColumnRole.CATEGORY_HIGH, ColumnRole.GEOGRAPHIC)]


def get_id_columns(schema: dict[str, ColumnProfile]) -> list[str]:
    return [p.name for p in schema.values() if p.role == ColumnRole.IDENTIFIER]
