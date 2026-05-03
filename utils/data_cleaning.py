import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df):
    """
    Performs basic preprocessing on the dataframe:
    - Handles missing values
    - Removes duplicates
    - Converts obvious date columns
    
    Returns:
        tuple: (cleaned_dataframe, summary_string)
    """
    df_clean = df.copy()
    initial_rows, initial_cols = df_clean.shape
    
    # 1. Remove duplicates
    duplicates_count = df_clean.duplicated().sum()
    if duplicates_count > 0:
        df_clean = df_clean.drop_duplicates()
        
    # 2. Handle missing values
    missing_count = df_clean.isna().sum().sum()
    numeric_imputed = 0
    cat_imputed = 0
    if missing_count > 0:
        # Numeric columns: fill with median (robust to outliers)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_missing = df_clean[col].isna().sum()
            if col_missing > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                numeric_imputed += col_missing
                
        # Categorical columns: fill with mode
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            col_missing = df_clean[col].isna().sum()
            if col_missing > 0:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")
                cat_imputed += col_missing
                    
    # 3. Detect and convert date columns
    date_converted_count = 0
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Simple heuristic: Check if column name suggests date or if values look like dates
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
                date_converted_count += 1
            except (ValueError, TypeError):
                pass
        else:
            # Try to infer if it's a date by testing first valid value
            first_valid = df_clean[col].dropna().iloc[0] if not df_clean[col].dropna().empty else None
            if isinstance(first_valid, str):
                # Basic check for typical date separators to avoid converting random strings
                if '-' in first_valid or '/' in first_valid:
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col])
                        date_converted_count += 1
                    except (ValueError, TypeError):
                        pass
                        
    # 4. Outlier Detection and Treatment (IQR Method for numeric columns)
    outliers_handled = 0
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        num_outliers = outlier_mask.sum()
        
        if num_outliers > 0:
            # Cap outliers at the boundaries
            df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
            df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
            outliers_handled += num_outliers

    # 5. Encoding Categorical Variables
    encoded_cols = 0
    # Select object or category columns that are not already converted to datetime
    cat_cols_to_encode = [col for col in df_clean.select_dtypes(include=['object', 'category']).columns 
                          if not pd.api.types.is_datetime64_any_dtype(df_clean[col])]
    
    le = LabelEncoder()
    for col in cat_cols_to_encode:
        # Only encode if number of unique values is reasonable (e.g., < 20) to avoid encoding free text
        if df_clean[col].nunique() < 20:
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            encoded_cols += 1

    # 6. Feature Scaling
    scaled_cols = 0
    # Re-evaluate numeric columns after potential encoding
    numeric_cols_to_scale = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols_to_scale) > 0:
        scaler = StandardScaler()
        df_clean[numeric_cols_to_scale] = scaler.fit_transform(df_clean[numeric_cols_to_scale])
        scaled_cols = len(numeric_cols_to_scale)

    # Build summary string
    summary = []
    if duplicates_count > 0 or missing_count > 0 or date_converted_count > 0:
        summary.append("Data Cleaning Techniques Applied:")
    if duplicates_count > 0:
        summary.append(f"- Removed {duplicates_count} duplicate rows to ensure data uniqueness.")
    if missing_count > 0:
        summary.append(f"- Imputed {missing_count} missing values across the dataset:")
        if numeric_imputed > 0:
            summary.append(f"  * Filled {numeric_imputed} missing numeric values with their respective column medians (robust to outliers).")
        if cat_imputed > 0:
            summary.append(f"  * Filled {cat_imputed} missing categorical values with their respective column modes or 'Unknown'.")
    if date_converted_count > 0:
        summary.append(f"- Converted {date_converted_count} column(s) to DateTime format using format inference.")
    if outliers_handled > 0:
        summary.append(f"- Cap-treated {outliers_handled} outliers across numeric columns using the Interquartile Range (IQR) method.")
    if encoded_cols > 0:
        summary.append(f"- Label Encoded {encoded_cols} categorical feature(s) containing fewer than 20 unique values.")
    if scaled_cols > 0:
        summary.append(f"- Standardized {scaled_cols} numeric feature(s) using Z-score scaling (StandardScaler) for uniform feature weight.")
        
    if not summary:
        summary_str = "No major cleaning needed. Data looks entirely clean!"
    else:
        summary_str = "\n".join(summary)
        
    return df_clean, summary_str
