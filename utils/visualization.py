import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

def plot_numeric_distributions(df):
    """
    Generate histograms and boxplots for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for distributions.")
        return
        
    selected_col = st.selectbox("Select Numeric Column to Visualize:", numeric_cols, key='num_dist_col')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(df, x=selected_col, title=f"Histogram of {selected_col}", 
                                marginal="violin", hover_data=df.columns)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        # Boxplot
        fig_box = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
        st.plotly_chart(fig_box, use_container_width=True)

def plot_categorical_distributions(df):
    """
    Generate bar charts for categorical columns.
    """
    cat_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
    
    if not cat_cols:
        st.warning("No categorical columns found.")
        return
        
    # Filter columns with too many unique values to prevent crowded charts
    valid_cat_cols = [col for col in cat_cols if df[col].nunique() < 50]
    
    if not valid_cat_cols:
        st.warning("All categorical columns have too many unique values (>50) for clear visualization.")
        return
        
    selected_col = st.selectbox("Select Categorical Column to Visualize:", valid_cat_cols, key='cat_dist_col')
    
    val_counts = df[selected_col].value_counts().reset_index()
    val_counts.columns = [selected_col, 'Count']
    
    # Sort or limit to top 20
    val_counts = val_counts.head(20)
    
    fig = px.bar(val_counts, x=selected_col, y='Count', 
                 title=f"Frequency Chart of {selected_col} (Top 20)",
                 color=selected_col)
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(df):
    """
    Generate correlation heatmap for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        st.warning("Not enough numeric columns to generate a correlation heatmap.")
        return
        
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1)
    
    # Adjust size
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_time_series(df):
    """
    Generate time series multi-line charts.
    """
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not date_cols:
        st.info("No datetime columns detected for time series analysis.")
        return
        
    if not numeric_cols:
        st.info("No numeric columns detected to plot over time.")
        return
        
    date_col = st.selectbox("Select Date Column:", date_cols, key='ts_date_col')
    val_col = st.selectbox("Select Value Column:", numeric_cols, key='ts_val_col')
    
    # Sort dataframe by date for proper time series plotting
    temp_df = df.sort_values(by=date_col).copy()
    
    fig = px.line(temp_df, x=date_col, y=val_col, 
                  title=f"{val_col} over {date_col}")
    st.plotly_chart(fig, use_container_width=True)
