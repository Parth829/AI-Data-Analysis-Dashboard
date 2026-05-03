import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def render_dashboard(df):
    """
    Renders the Dataset Summary Dashboard showing:
    - Number of rows & columns
    - Data types
    - Missing values distribution
    """
    st.header("Dataset Summary Dashboard")
    
    # 1. High-level metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Rows", value=f"{df.shape[0]:,}")
    with col2:
        st.metric(label="Total Columns", value=f"{df.shape[1]:,}")
    with col3:
        missing_cells = df.isna().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_percent = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        st.metric(label="Missing Cells", value=f"{missing_cells:,} ({missing_percent:.1f}%)")
        
    st.markdown("---")
    
    col_dist, col_miss = st.columns(2)
    
    with col_dist:
        st.subheader("Data Type Distribution")
        dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        
        fig = px.pie(dtype_counts, values='Count', names='Data Type', hole=0.4, 
                     title="Columns by Data Type")
        st.plotly_chart(fig, use_container_width=True)
        
    with col_miss:
        st.subheader("Missing Values per Column")
        missing_data = df.isna().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        missing_data = missing_data[missing_data['Missing Values'] > 0]
        
        if not missing_data.empty:
            fig = px.bar(missing_data, x='Column', y='Missing Values', 
                         title="Missing Values Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values in the dataset! 🎉")
            
    st.markdown("---")
    st.subheader("Column Statistics Overview")
    
    # Show describe for numeric types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write("**Numeric Columns Statistics:**")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        
    # Show basic info for categorical types
    cat_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns
    if len(cat_cols) > 0:
        st.write("**Categorical Columns Statistics:**")
        st.dataframe(df[cat_cols].describe().T, use_container_width=True)
