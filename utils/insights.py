import pandas as pd
import numpy as np
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

def generate_insights(df):
    """
    Analyzes the dataframe and returns a list of text-based business insights.
    """
    insights = []
    
    # 1. Dataset Shape Insights
    insights.append(f"**Dataset Size**: You have a dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns.")
    
    # 2. Missing Value Insights
    missing_sum = df.isna().sum().sum()
    if missing_sum > 0:
        col_most_missing = df.isna().sum().idxmax()
        missing_count = df.isna().sum().max()
        insights.append(f"**Data Completeness**: There were missing values. The column '{col_most_missing}' had the highest number of missing values ({missing_count:,}).")
    else:
        insights.append("**Data Completeness**: Your dataset has no missing values.")
        
    # 3. Numeric Column Insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Find column with highest variance/spread relative to its mean
        coef_of_var = {}
        for col in numeric_cols:
            mean = df[col].mean()
            if mean != 0:
                cv = df[col].std() / mean
                coef_of_var[col] = cv
                
        if coef_of_var:
            most_volatile = max(coef_of_var, key=coef_of_var.get)
            insights.append(f"**Volatility**: Among numeric features, '{most_volatile}' shows the highest relative variability.")
            
        # 4. Correlation Insights
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Find the highest absolute correlation (excluding self-correlation trick)
            corr_matrix.values[[np.arange(corr_matrix.shape[0])]*2] = np.nan
            
            # Find max pos/neg corr
            try:
                max_corr_idx = corr_matrix.abs().unstack().idxmax()
                col1, col2 = max_corr_idx
                corr_val = corr_matrix.loc[col1, col2]
                
                if abs(corr_val) > 0.7:
                    strength = "strong"
                elif abs(corr_val) > 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"
                    
                direction = "positive" if corr_val > 0 else "negative"
                insights.append(f"**Correlations**: '{col1}' and '{col2}' have a {strength} {direction} correlation ({corr_val:.2f}).")
            except Exception:
                pass

    # 5. Categorical Insights
    cat_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns
    if len(cat_cols) > 0:
        # Find categorical column with fewest unique values to show dominance
        valid_cat_cols = [col for col in cat_cols if 1 < df[col].nunique() < 20]
        if valid_cat_cols:
            for col in valid_cat_cols[:2]: # Show up to 2
                top_val = df[col].value_counts().idxmax()
                top_pct = (df[col].value_counts().max() / len(df)) * 100
                if top_pct > 50:
                    insights.append(f"**Dominant Category**: In '{col}', '{top_val}' dominates the dataset, making up {top_pct:.1f}% of all records.")

    return insights

def generate_ai_insights(df):
    """
    Uses Gemini via LangChain to generate advanced business insights
    based on the dataframe's statistical summary.
    API key is resolved internally via get_api_key() — no UI parameter.
    """
    from config.settings import get_api_key
    api_key = get_api_key()
        
    if not api_key:
        return ["AI insights unavailable — configure GOOGLE_API_KEY in your environment."]
        
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)
        
        # Create a summary of the dataset to send to the LLM
        # We don't send the whole dataset, just the summary stats to save tokens and maintain privacy
        
        buffer = []
        buffer.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Get basic info
        buffer.append("\nColumn Data Types:")
        for col, dtype in df.dtypes.items():
            buffer.append(f"- {col}: {dtype}")
            
        # Get summary statistics for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            buffer.append("\nNumeric Summary Columns Statistics:")
            buffer.append(numeric_df.describe().to_string())
            
        # Get top values for categorical columns
        cat_df = df.select_dtypes(exclude=[np.number, 'datetime'])
        if not cat_df.empty:
            buffer.append("\nCategorical Top Values:")
            for col in cat_df.columns:
                top_vals = df[col].value_counts().head(3)
                buffer.append(f"- {col}: {top_vals.to_dict()}")
                
        data_summary = "\n".join(buffer)
        
        messages = [
            SystemMessage(content="You are an expert data analyst. Your task is to review the following statistical summary of a dataset and provide exactly 3-5 high-value, actionable business insights. Focus on interesting patterns, potential anomalies, or key takeaways. Format your response as a bulleted list using markdown. Do not include introductory or concluding fluff."),
            HumanMessage(content=f"Here is the dataset summary:\n\n{data_summary}")
        ]
        
        response = llm.invoke(messages)
        
        # Parse the response into a list of strings
        insights_text = response.content
        ai_insights = [line.strip() for line in insights_text.split('\n') if line.strip().startswith('-') or line.strip().startswith('*')]
        
        if not ai_insights:
            # Fallback if the LLM didn't use bullet points
            ai_insights = [insights_text]
            
        return ai_insights
        
    except Exception as e:
        return [f"**Error** generating AI insights: {str(e)}"]

