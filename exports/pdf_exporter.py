"""
PDF exporter — fpdf2 with Unicode support. Chart images optional (kaleido).
"""
import datetime
from io import BytesIO
from fpdf import FPDF

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "AI-Powered Data Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "I", 10)
        self.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(79, 70, 229)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def section_body(self, text):
        self.set_font("Helvetica", "", 10)
        # fpdf2 handles Unicode natively
        self.multi_cell(0, 6, text)
        self.ln(4)

    def add_chart_image(self, fig, width=180):
        """Add a Plotly chart as image (requires kaleido)."""
        if not KALEIDO_AVAILABLE:
            return
        try:
            import copy
            export_fig = copy.deepcopy(fig)
            # Ensure labels aren't clipped: generous left margin + white bg
            export_fig.update_layout(
                margin=dict(l=140, r=40, t=60, b=60),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(size=12),
            )
            img_bytes = export_fig.to_image(format="png", width=1200, height=500, scale=2)
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(img_bytes)
            tmp.close()
            # Check for page overflow — add new page if not enough space
            if self.get_y() + 90 > self.h - 20:
                self.add_page()
            self.image(tmp.name, w=width)
            os.unlink(tmp.name)
            self.ln(6)
        except Exception:
            pass


def export_to_pdf(cleaning_summary, stat_insights, ai_insights, figures=None):
    """
    Generate PDF report.
    - cleaning_summary: string
    - stat_insights: list of Insight objects or strings
    - ai_insights: list of dicts or strings
    - figures: optional list of Plotly figures to embed
    Returns bytes.
    """
    pdf = PDFReport()
    pdf.add_page()

    # Cleaning summary
    pdf.section_title("Data Cleaning Summary")
    pdf.section_body(cleaning_summary if isinstance(cleaning_summary, str) else "\n".join(str(a) for a in cleaning_summary))

    # Statistical insights
    pdf.section_title("Statistical Insights")
    if stat_insights:
        lines = []
        for ins in stat_insights:
            if hasattr(ins, "title"):
                lines.append(f"[{ins.severity.upper()}] {ins.title}: {ins.description}")
            else:
                lines.append(str(ins).replace("**", ""))
        pdf.section_body("\n".join(lines))
    else:
        pdf.section_body("No statistical insights generated.")

    # AI insights — always include section
    pdf.section_title("AI-Generated Insights (Gemini)")
    if ai_insights:
        lines = []
        for item in ai_insights:
            if isinstance(item, dict):
                lines.append(f"- {item.get('insight', '')} | {item.get('recommendation', '')}")
            else:
                lines.append(str(item).replace("**", ""))
        pdf.section_body("\n".join(lines))
    else:
        pdf.section_body("No AI insights generated.")

    # Chart images — always include section
    pdf.add_page()
    pdf.section_title("Charts")
    if figures and KALEIDO_AVAILABLE:
        for fig in figures[:6]:  # limit to 6 charts
            pdf.add_chart_image(fig)
    else:
        if not KALEIDO_AVAILABLE:
            pdf.section_body("Charts could not be included because the 'kaleido' package is not installed.")
        else:
            pdf.section_body("No charts available for this dataset.")

    return bytes(pdf.output())
