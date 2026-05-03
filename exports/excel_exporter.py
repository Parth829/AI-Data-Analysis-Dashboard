"""
Excel exporter — multiple sheets: Raw, Cleaned, Stats, Insights, Anomalies.
"""
import pandas as pd
from io import BytesIO


def export_to_excel(df_raw, df_clean, insights=None, anomalies=None):
    """
    Export data to Excel with multiple sheets.
    Returns bytes.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Raw data
        if df_raw is not None:
            df_raw.to_excel(writer, index=False, sheet_name="Raw Data")

        # Cleaned data (exclude outlier flag columns)
        if df_clean is not None:
            display_cols = [c for c in df_clean.columns if not c.startswith("_is_outlier_")]
            df_clean[display_cols].to_excel(writer, index=False, sheet_name="Cleaned Data")

        # Summary stats
        if df_clean is not None:
            display_cols = [c for c in df_clean.columns if not c.startswith("_is_outlier_")]
            try:
                stats = df_clean[display_cols].describe(include="all")
                stats.to_excel(writer, sheet_name="Summary Stats")
            except Exception:
                pass

        # Insights
        if insights:
            insights_data = []
            for ins in insights:
                if hasattr(ins, "title"):
                    insights_data.append({
                        "Title": ins.title,
                        "Severity": ins.severity,
                        "Description": ins.description,
                        "Recommendation": ins.recommendation,
                    })
                elif isinstance(ins, dict):
                    insights_data.append(ins)
            if insights_data:
                pd.DataFrame(insights_data).to_excel(writer, index=False, sheet_name="Insights")

        # Anomalies
        if anomalies is not None and len(anomalies) > 0:
            anomalies.to_excel(writer, index=False, sheet_name="Anomalies")

    return output.getvalue()
