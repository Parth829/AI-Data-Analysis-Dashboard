"""
File loading with encoding detection, format support, and caching.
Supports: CSV, XLSX, XLS, Parquet, JSON.
"""
import hashlib
import pandas as pd
import streamlit as st
import chardet
from config.settings import LARGE_FILE_THRESHOLD_MB, SAMPLE_SIZES


def _detect_encoding(file_bytes: bytes) -> str:
    """Detect file encoding using chardet."""
    result = chardet.detect(file_bytes[:100_000])  # sample first 100KB
    encoding = result.get("encoding", "utf-8") or "utf-8"
    return encoding


def _compute_file_hash(file_bytes: bytes) -> str:
    """Deterministic cache key from file contents."""
    return hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def _load_cached(file_hash: str, file_bytes: bytes, file_name: str,
                 sample_size: int | None = None) -> pd.DataFrame:
    """
    Internal cached loader. Keyed on file_hash so identical files hit cache.
    """
    ext = file_name.rsplit(".", 1)[-1].lower()

    if ext == "csv":
        encoding = _detect_encoding(file_bytes)
        try:
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError:
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_bytes), encoding="latin-1")

    elif ext in ("xlsx", "xls"):
        from io import BytesIO
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl" if ext == "xlsx" else None)

    elif ext == "parquet":
        from io import BytesIO
        df = pd.read_parquet(BytesIO(file_bytes))

    elif ext == "json":
        from io import BytesIO
        df = pd.read_json(BytesIO(file_bytes))

    else:
        raise ValueError(f"Unsupported file format: .{ext}")

    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


def load_file(uploaded_file) -> pd.DataFrame | None:
    """
    Load an uploaded file into a DataFrame.
    Handles large-file sampling via a UI radio button.
    Returns None on failure.
    """
    if uploaded_file is None:
        return None

    try:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_hash = _compute_file_hash(file_bytes)
        file_size_mb = len(file_bytes) / (1024 * 1024)

        # Large file → offer sampling
        sample_size = None
        if file_size_mb > LARGE_FILE_THRESHOLD_MB:
            st.warning(f"File is large ({file_size_mb:.0f} MB). Consider sampling for faster analysis.")
            choice = st.radio(
                "Load option:",
                ["Load all rows", f"Sample {SAMPLE_SIZES['50k']:,} rows", f"Sample {SAMPLE_SIZES['10k']:,} rows"],
                key="load_sample_option",
            )
            if "50k" in choice:
                sample_size = SAMPLE_SIZES["50k"]
            elif "10k" in choice:
                sample_size = SAMPLE_SIZES["10k"]

        df = _load_cached(file_hash, file_bytes, file_name, sample_size)
        return df

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


def load_sample_data(dataset_name: str) -> pd.DataFrame:
    """Load one of the bundled sample datasets."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "sample_data", dataset_name)
    return pd.read_csv(path)
