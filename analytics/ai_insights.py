"""
Gemini AI insights — plain function, no ABC.
Structured prompt with domain/goal hints and JSON output.
"""
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st
from config.settings import GEMINI_MODEL, get_api_key
from core.schema_profiler import ColumnProfile, get_numeric_columns, get_category_columns, get_date_columns


def _build_data_summary(df, schema):
    """Build a concise data summary for the LLM prompt."""
    display_cols = [c for c in df.columns if not c.startswith("_is_outlier_")]
    df_d = df[display_cols]
    parts = [f"Dataset: {df_d.shape[0]} rows × {df_d.shape[1]} columns"]

    # Column roles
    parts.append("\nColumn roles:")
    for col, prof in schema.items():
        parts.append(f"  - {col}: {prof.role.value} ({prof.dtype})")

    # Numeric summary
    num_cols = [c for c in get_numeric_columns(schema) if c in df_d.columns]
    if num_cols:
        parts.append("\nNumeric summary:")
        parts.append(df_d[num_cols].describe().round(2).to_string())

    # Top correlations
    if len(num_cols) >= 2:
        corr = df_d[num_cols].corr()
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                pairs.append((num_cols[i], num_cols[j], corr.iloc[i, j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if pairs:
            parts.append("\nTop correlations:")
            for c1, c2, v in pairs[:5]:
                parts.append(f"  {c1} ↔ {c2}: {v:.3f}")

    # Category breakdown
    cat_cols = [c for c in get_category_columns(schema) if c in df_d.columns]
    for col in cat_cols[:3]:
        vc = df_d[col].value_counts().head(5)
        parts.append(f"\n{col} distribution: {vc.to_dict()}")

    # Outlier count
    outlier_cols = [c for c in df.columns if c.startswith("_is_outlier_")]
    if outlier_cols:
        total_outliers = sum(df[c].sum() for c in outlier_cols)
        parts.append(f"\nTotal flagged outliers: {total_outliers}")

    return "\n".join(parts)


def generate_gemini_insights(df, schema, domain="General", goal="General analysis"):
    """
    Generate AI insights using Gemini.
    Returns list of dicts [{insight, evidence, confidence, recommendation}] or None on failure.
    Cached by data summary + domain + goal.
    The API key is resolved internally via get_api_key() — no UI parameter.
    """
    api_key = get_api_key()
    if not api_key:
        return None

    data_summary = _build_data_summary(df, schema)

    # Cache check
    cache_key = hashlib.md5((data_summary + domain + goal).encode()).hexdigest()
    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = {}
    if cache_key in st.session_state["ai_cache"]:
        return st.session_state["ai_cache"][cache_key]

    system_prompt = f"""You are an expert data analyst. Your specific industry domain is '{domain}', and your primary objective for this analysis is to '{goal}'.

Analyze the dataset summary below strictly through the lens of this domain and goal. Every single insight you generate MUST be directly relevant to achieving '{goal}' in a '{domain}' context. Avoid generic observations.

Return ONLY a JSON array where each element has these fields:
- "insight": one-line finding directly tied to the goal (string)
- "evidence": specific numbers/data supporting it (string)
- "confidence": "high", "medium", or "low" (string)
- "recommendation": highly actionable next step tailored to {domain} (string)

Be specific with numbers. Reference actual column names and values from the data.
Do not include any text outside the JSON array."""

    user_prompt = f"Please analyze this data to '{goal}' for the '{domain}' industry.\n\nDataset summary:\n{data_summary}"

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.3,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        # Parse JSON from response
        text = response.content.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]

        insights = json.loads(text)
        if isinstance(insights, list):
            st.session_state["ai_cache"][cache_key] = insights
            return insights

        return None

    except json.JSONDecodeError:
        # Fallback: try to extract bullet points
        try:
            lines = response.content.strip().split("\n")
            insights = [{"insight": l.strip("- *"), "evidence": "", "confidence": "medium", "recommendation": ""}
                        for l in lines if l.strip() and (l.strip().startswith("-") or l.strip().startswith("*"))]
            if insights:
                st.session_state["ai_cache"][cache_key] = insights
                return insights
        except Exception:
            pass
        return None
    except Exception as e:
        # Surface the error so the UI can display it
        error_msg = str(e)
        recommendation = "Check your API key and network connection."
        
        # Common error patterns — provide actionable messages
        if "503" in error_msg or "high demand" in error_msg.lower():
            error_msg = "Google's Gemini servers are currently experiencing high demand."
            recommendation = "This is a temporary issue on Google's end. Please wait a few minutes and try again."
        elif "API_KEY_INVALID" in error_msg or "401" in error_msg:
            error_msg = "Invalid API key. Verify GOOGLE_API_KEY in your .env file."
        elif "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            error_msg = "API quota exceeded. Wait a moment and try again."
        elif "PERMISSION_DENIED" in error_msg or "403" in error_msg:
            error_msg = "API key does not have permission for this model. Check your Google AI Studio settings."
            
        return [{"insight": f"⚠️ Gemini API error: {error_msg}", "evidence": "", "confidence": "low", "recommendation": recommendation}]

