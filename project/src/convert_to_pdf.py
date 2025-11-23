"""
PDF Conversion Script for Final Report
Uses markdown2 and weasyprint to convert final_report.md to PDF with images
"""

import markdown2
from weasyprint import HTML, CSS
from pathlib import Path
import re

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "REPORT"
RESULTS_DIR = PROJECT_ROOT / "results"

# Read the markdown file
md_file = REPORT_DIR / "final_report.md"
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML with extensions
html_content = markdown2.markdown(
    md_content,
    extras=["tables", "fenced-code-blocks", "code-friendly", "break-on-newline"]
)

# Fix relative image paths to absolute paths
html_content = html_content.replace('src="../results/', f'src="file:///{RESULTS_DIR}/')
html_content = html_content.replace('src="..\\results\\', f'src="file:///{RESULTS_DIR}/')

# Create full HTML document with styling
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @page {{
            size: A4;
            margin: 1.5cm 2cm;
            @bottom-center {{
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #666;
            }}
        }}
        
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }}
        
        h1 {{
            font-size: 24pt;
            color: #1a1a1a;
            border-bottom: 3px solid #4a90e2;
            padding-bottom: 10px;
            margin-top: 20px;
            page-break-after: avoid;
        }}
        
        h2 {{
            font-size: 18pt;
            color: #2c3e50;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-top: 25px;
            page-break-after: avoid;
        }}
        
        h3 {{
            font-size: 14pt;
            color: #34495e;
            margin-top: 20px;
            page-break-after: avoid;
        }}
        
        h4 {{
            font-size: 12pt;
            color: #555;
            margin-top: 15px;
            page-break-after: avoid;
        }}
        
        p {{
            text-align: justify;
            margin: 10px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }}
        
        th {{
            background-color: #4a90e2;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: bold;
        }}
        
        td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
        }}
        
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-left: 4px solid #4a90e2;
            padding: 15px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            page-break-inside: avoid;
        }}
        
        ul, ol {{
            margin: 10px 0 10px 25px;
        }}
        
        li {{
            margin: 5px 0;
        }}
        
        blockquote {{
            border-left: 4px solid #4a90e2;
            padding-left: 15px;
            margin: 15px 0;
            color: #666;
            font-style: italic;
        }}
        
        a {{
            color: #4a90e2;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .page-break {{
            page-break-before: always;
        }}
        
        strong {{
            font-weight: bold;
            color: #1a1a1a;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Output PDF path
output_pdf = REPORT_DIR / "final_report.pdf"

# Convert HTML to PDF
print(f"Converting {md_file} to PDF...")
print(f"Output: {output_pdf}")

HTML(string=full_html, base_url=str(REPORT_DIR)).write_pdf(
    output_pdf,
    stylesheets=None,
    presentational_hints=True
)

print(f"\nPDF conversion complete!")
print(f"File saved: {output_pdf}")
print(f"File size: {output_pdf.stat().st_size / 1024:.2f} KB")
