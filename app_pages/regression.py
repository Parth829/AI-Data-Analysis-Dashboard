"""
Regression page — full evaluation with R², MAE, MAPE, RMSE, residuals, cross-validation.
"""
import streamlit as st
import numpy as np
import pandas as pd
from core.schema_profiler import get_numeric_columns, get_category_columns, get_id_columns
from core.data_cleaner import get_display_columns
from ml.regression import run_regression
from ml.model_evaluation import plot_cv_results
from visualizations.chart_themes import apply_theme


def render_regression(df, schema):
    """Render the regression page."""
    st.header("Regression Analysis")

    display_cols = get_display_columns(df)
    num_cols = get_numeric_columns(schema)
    cat_cols = get_category_columns(schema)
    id_cols = get_id_columns(schema)

    # Available features: numeric + categorical, excluding IDs
    feature_options = [c for c in num_cols + cat_cols if c not in id_cols and c in df.columns]

    if len(num_cols) < 1:
        st.warning("Need at least 1 numeric column for regression.")
        return

    # Target selection (must be numeric)
    target = st.selectbox("Target variable (Y):", num_cols, key="reg_target")
    features = st.multiselect(
        "Feature variables (X):",
        [c for c in feature_options if c != target],
        default=[c for c in feature_options if c != target][:5],
        key="reg_features",
    )

    if not features:
        st.info("Select at least one feature variable.")
        return

    cat_in_features = [f for f in features if f in cat_cols]
    if cat_in_features:
        st.caption(f"Categorical features will be one-hot encoded internally: {', '.join(cat_in_features)}")

    if st.button("Run Regression", key="run_regression"):
        with st.spinner("Training Random Forest..."):
            result = run_regression(df, target, features, schema)

        if result.get("error_msg"):
            st.error(result["error_msg"])
            return

        # Metrics comparison
        st.subheader("Model Performance")
        st.caption("Compare train vs test to check for overfitting.")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Train Metrics**")
            for k, v in result["train_metrics"].items():
                st.metric(k, f"{v:.4f}")
        with c2:
            st.write("**Test Metrics**")
            for k, v in result["test_metrics"].items():
                st.metric(k, f"{v:.4f}")

        st.markdown("---")

        # Plots
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Actual vs Predicted")
            fig = result["fig_actual_pred"]
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            st.subheader("Residuals")
            fig = result["fig_residuals"]
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Feature importances
        st.subheader("Feature Importances")
        imp = result["importances"]
        import plotly.express as px
        fig = px.bar(imp.head(15), x="Importance", y="Feature", orientation="h",
                     title="Top 15 Features")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Cross-validation
        if result.get("cv_summary"):
            st.markdown("---")
            st.subheader("Cross-Validation (5-Fold)")
            cv_df = pd.DataFrame([result["cv_summary"]], index=["Mean ± Std"])
            st.dataframe(cv_df, use_container_width=True)

            if result.get("cv_metrics"):
                fig = plot_cv_results(result["cv_metrics"])
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
