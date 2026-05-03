"""
Custom chart builder — user selects axes and chart type.
"""
import streamlit as st
import plotly.express as px
from visualizations.chart_themes import apply_theme


CHART_TYPES = [
    "Bar", "Line", "Scatter", "BoxPlot", "Histogram",
    "Pie", "Area", "Violin", "Density Heatmap", "Strip", "Funnel",
]


def render_custom_chart_builder(df):
    """Render the custom chart builder UI."""
    st.subheader("🛠️ Build Your Own Chart")

    col_x, col_y, col_chart = st.columns(3)
    with col_x:
        x_axis = st.selectbox("X-Axis", df.columns, key="custom_x")
    with col_y:
        y_axis = st.selectbox("Y-Axis", df.columns, key="custom_y")
    with col_chart:
        chart_type = st.selectbox("Chart Type", CHART_TYPES, key="custom_chart")

    if st.button("Generate Chart", key="gen_custom_chart"):
        try:
            fig = _create_chart(df, x_axis, y_axis, chart_type)
            if fig:
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate chart: {str(e)}")


def _create_chart(df, x, y, chart_type):
    """Create a Plotly figure for the given chart type."""
    title = f"{chart_type}: {y} by {x}"

    if chart_type == "Bar":
        agg = df.groupby(x)[y].sum().reset_index()
        return px.bar(agg, x=x, y=y, title=title)

    if chart_type == "Line":
        return px.line(df.sort_values(x), x=x, y=y, title=title)

    if chart_type == "Scatter":
        return px.scatter(df, x=x, y=y, title=title)

    if chart_type == "BoxPlot":
        return px.box(df, x=x, y=y, title=title)

    if chart_type == "Histogram":
        return px.histogram(df, x=x, y=y, title=title)

    if chart_type == "Pie":
        agg = df.groupby(x)[y].sum().reset_index()
        return px.pie(agg, names=x, values=y, title=title)

    if chart_type == "Area":
        agg = df.groupby(x)[y].sum().reset_index()
        return px.area(agg, x=x, y=y, title=title)

    if chart_type == "Violin":
        return px.violin(df, x=x, y=y, title=title)

    if chart_type == "Density Heatmap":
        return px.density_heatmap(df, x=x, y=y, title=title)

    if chart_type == "Strip":
        return px.strip(df, x=x, y=y, title=title)

    if chart_type == "Funnel":
        agg = df.groupby(x)[y].sum().reset_index()
        return px.funnel(agg, x=x, y=y, title=title)

    return None
