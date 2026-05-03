"""
UI Themes and CSS injection.
Target: Streamlit >= 1.35
"""
import streamlit as st
import traceback

def inject_css():
    """Inject global CSS for FAANG-level visual polish, adapting to light/dark mode."""
    
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

    :root {
        --accent: #4F46E5;
        --accent-soft: color-mix(in srgb, var(--accent) 15%, transparent);
        --border-subtle: color-mix(in srgb, var(--text-color) 10%, transparent);
        --border-strong: color-mix(in srgb, var(--text-color) 25%, transparent);
        --text-muted: color-mix(in srgb, var(--text-color) 60%, transparent);
        --bg-hover: color-mix(in srgb, var(--text-color) 5%, transparent);
    }

    /* Global Typography */
    html, body, [class*="st-"]:not([data-testid="stIconMaterial"]) {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    }

    /* Explicitly let Material icon containers use their icon font */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    /* Hide Streamlit Chrome */
    /* Streamlit version: 1.52.2 uses [data-testid="stSidebarCollapsedControl"] */
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    .stDeployButton, [data-testid="stDeployButton"] { display: none !important; }

    /* DO NOT hide the entire header — it contains the sidebar toggle.
       Just make it transparent so it doesn't visually clash. */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: auto !important;
    }

    /* Explicitly ensure both sidebar toggles remain visible */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }



    /* Layout Spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1400px !important;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border-subtle) !important;
    }
    .sidebar-section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* Headers and Text */
    h1 {
        font-size: 28px !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-color) !important;
        margin-bottom: 0.25rem !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 14px !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid var(--border-strong) !important;
        background: var(--background-color) !important;
        color: var(--text-color) !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* Radio Nav (Sidebar Menu) */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 2px !important;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"] {
        padding: 8px 12px !important;
        border-radius: 6px !important;
        transition: background 0.15s ease !important;
        cursor: pointer !important;
        margin-bottom: 2px !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
        background: var(--bg-hover) !important;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
        background: var(--accent-soft) !important;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p {
        color: var(--accent) !important;
        font-weight: 600 !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed var(--border-strong) !important;
        border-radius: 10px !important;
        background: var(--secondary-background-color) !important;
        padding: 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    /* Multiselect Tags */
    span[data-baseweb="tag"] {
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
        border-radius: 6px !important;
    }

    /* Utilities */
    .dash-spacer-sm { height: 16px; }
    .dash-spacer-md { height: 32px; }
    .dash-spacer-lg { height: 64px; }

    /* Custom Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .app-header-left .app-title {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-color);
        letter-spacing: -0.02em;
    }
    .app-header-left .app-subtitle {
        font-size: 14px;
        color: var(--text-muted);
        margin-top: 4px;
    }
    .app-status-pill {
        font-size: 13px;
        font-weight: 500;
        padding: 6px 12px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .app-status-pill.inactive {
        background: var(--bg-hover);
        color: var(--text-muted);
    }
    .app-status-pill.active {
        background: var(--background-color);
        border: 1px solid var(--border-subtle);
        color: var(--text-color);
    }
    .app-status-pill.active::before {
        content: "";
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
    }

    /* Landing Page Feature Cards */
    .feature-card {
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        background: var(--secondary-background-color);
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border-color: var(--border-strong);
    }
    .feature-icon {
        color: var(--accent);
        margin-bottom: 16px;
    }
    .feature-title {
        font-weight: 600;
        font-size: 16px;
        color: var(--text-color);
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 14px;
        color: var(--text-muted);
        line-height: 1.5;
    }
    .format-pill {
        background: var(--bg-hover);
        color: var(--text-muted);
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        display: inline-block;
        margin-right: 6px;
    }
    .inline-status-pill {
        background: var(--bg-hover);
        color: var(--text-muted);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        display: inline-block;
        margin-top: 8px;
    }

    /* Landing Page Hero Text */
    .landing-hero-title {
        font-size: 32px; 
        font-weight: 700; 
        color: var(--text-color); 
        margin-bottom: 12px;
    }
    .landing-hero-desc {
        font-size: 16px; 
        color: var(--text-muted); 
        line-height: 1.5;
    }

    /* Styled Error */
    .error-card {
        border: 1px solid #EF4444;
        border-radius: 8px;
        padding: 16px;
        background: color-mix(in srgb, #EF4444 10%, transparent);
        color: #EF4444;
        margin-bottom: 16px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .error-card-title {
        font-weight: 600; 
        margin-bottom: 4px;
    }
    .error-card-msg {
        font-size: 14px; 
        opacity: 0.9;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def sidebar_section(label: str):
    """Render a styled uppercase section label in the sidebar."""
    st.sidebar.markdown(f"<div class='sidebar-section-label'>{label}</div>", unsafe_allow_html=True)


def render_app_header(filename: str):
    """Render the custom top header bar with status pill."""
    if filename:
        pill_html = f'<div class="app-status-pill active">{filename}</div>'
    else:
        pill_html = '<div class="app-status-pill inactive">No data loaded</div>'

    html = f"""
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-title">Data Analysis</div>
            <div class="app-subtitle">AI-powered insights, visualizations, and forecasts</div>
        </div>
        <div class="app-header-right">
            {pill_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_styled_error(error: Exception):
    """Render a polished error card with a traceback expander."""
    err_html = f"""
    <div class="error-card">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <div>
            <div class="error-card-title">Something went wrong</div>
            <div class="error-card-msg">{str(error)}</div>
        </div>
    </div>
    """
    st.markdown(err_html, unsafe_allow_html=True)
    with st.expander("Show error details"):
        st.code(traceback.format_exc(), language="python")


# Lucide SVG strings for landing page
SVG_BAR_CHART = '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'''
SVG_SPARKLES = '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"></path></svg>'''
SVG_TRENDING_UP = '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg>'''
