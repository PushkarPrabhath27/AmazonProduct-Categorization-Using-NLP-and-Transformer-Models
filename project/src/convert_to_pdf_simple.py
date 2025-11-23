"""
Simpler PDF conversion using markdown-pdf library
"""
import pypandoc
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "REPORT"

# Input and output paths
md_file = str(REPORT_DIR / "final_report.md")
output_pdf = str(REPORT_DIR / "final_report.pdf")

print(f"Converting {md_file} to PDF using pandoc...")

# Convert to PDF with pandoc
# This will preserve images and formatting
pypandoc.convert_file(
    md_file,
    'pdf',
    outputfile=output_pdf,
    extra_args=[
        '--pdf-engine=pdflatex',
        '--variable=geometry:margin=1in',
        '--variable=fontsize:11pt',
        '--variable=mainfont:Georgia',
        '--toc',
        '--toc-depth=3',
        '-V', 'linkcolor:blue',
        '-V', 'urlcolor:blue',  
        '--highlight-style=tango'
    ],
    sandbox=False
)

print(f"\nPDF created: {output_pdf}")
pdf_path = Path(output_pdf)
if pdf_path.exists():
    print(f"File size: {pdf_path.stat().st_size / 1024:.2f} KB")
    print(f"Success!")
