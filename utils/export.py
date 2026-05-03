import pandas as pd
from io import BytesIO
from fpdf import FPDF
import datetime

def export_to_excel(df):
    """
    Exports a dataframe to an Excel file in memory (BytesIO).
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        # Add a sheet with basic stats
        if not df.empty:
             stats = df.describe(include='all')
             stats.to_excel(writer, sheet_name='Summary Stats')
    processed_data = output.getvalue()
    return processed_data

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI-Powered Data Analysis Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        # Using multi_cell to handle line breaks
        # Clean text basic
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 8, text)
        self.ln()

def export_to_pdf(cleaning_summary, auto_insights, ai_insights):
    """
    Generates a PDF report containing summaries and insights.
    Returns the raw bytes of the PDF.
    """
    pdf = PDFReport()
    pdf.add_page()
    
    # Cleaning Summary
    pdf.chapter_title('Data Cleaning Summary')
    pdf.chapter_body(cleaning_summary)
    
    # Statistical Insights
    pdf.chapter_title('Statistical Business Insights')
    if auto_insights:
        stat_text = "\n".join([f"- {insight.replace('**', '')}" for insight in auto_insights])
    else:
        stat_text = "No statistical insights generated."
    pdf.chapter_body(stat_text)
    
    # AI Insights
    pdf.chapter_title('AI-Generated Insights (OpenAI/LangChain)')
    if ai_insights:
         ai_text = "\n".join([insight.replace('**', '') for insight in ai_insights])
    else:
         ai_text = "No AI insights generated (API key may be missing)."
    pdf.chapter_body(ai_text)
    
    # Save to BytesIO
    # Note: FPDF2 allows output dest='S' to return string/bytes
    try:
         pdf_bytes = pdf.output(dest='S').encode('latin-1')
    except Exception:
         # Fallback for FPDF PyFPDF 1.7
         pdf_bytes = pdf.output(dest='S')
         if type(pdf_bytes) == str:
              pdf_bytes = pdf_bytes.encode('latin-1')
              
    return pdf_bytes
