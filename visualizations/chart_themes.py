"""
Consistent Plotly theme and styling for all charts.
"""
import plotly.graph_objects as go
import plotly.io as pio

# Color palette
COLORS = {
    # Existing (preserved)
    "primary": "#4F46E5",     # indigo
    "secondary": "#7C3AED",   # violet
    "success": "#10B981",     # emerald
    "warning": "#F59E0B",     # amber
    "danger": "#EF4444",      # red
    "info": "#3B82F6",        # blue
    "muted": "#6B7280",       # gray
    # New tokens for UI theme
    "accent": "#4F46E5",
    "accent_soft": "#EEF2FF",
    "text_primary": "#111827",
    "text_secondary": "#374151",
    "text_muted": "#6B7280",
    "bg_primary": "#FFFFFF",
    "bg_secondary": "#FAFAFA",
    "bg_tertiary": "#F3F4F6",
    "border": "#E5E7EB",
    "border_strong": "#D1D5DB",
}

COLOR_SEQUENCE = [
    "#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981",
    "#3B82F6", "#EF4444", "#8B5CF6", "#14B8A6", "#F97316",
]

# Register custom template
TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#374151"),
        title=dict(font=dict(size=18, color="#111827"), x=0, xanchor="left"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=COLOR_SEQUENCE,
        xaxis=dict(gridcolor="#E5E7EB", gridwidth=1, linecolor="#D1D5DB"),
        yaxis=dict(gridcolor="#E5E7EB", gridwidth=1, linecolor="#D1D5DB"),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
)

pio.templates["dashboard"] = TEMPLATE
pio.templates.default = "dashboard"


def apply_theme(fig):
    """Apply the dashboard theme to a figure."""
    fig.update_layout(template="dashboard")
    return fig


def severity_color(severity):
    """Map insight severity to color."""
    return {"critical": COLORS["danger"], "warning": COLORS["warning"], "info": COLORS["info"]}.get(severity, COLORS["muted"])


def quality_color(score):
    """Map quality score (0-100) to color."""
    if score >= 90:
        return COLORS["success"]
    if score >= 70:
        return COLORS["warning"]
    return COLORS["danger"]
