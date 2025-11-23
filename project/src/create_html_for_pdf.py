"""
Convert markdown to HTML with embedded images for easy PDF printing
"""
import markdown2
from pathlib import Path
import base64

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "REPORT"
RESULTS_DIR = PROJECT_ROOT / "results"

# Read the markdown file
md_file = REPORT_DIR / "final_report.md"
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_body = markdown2.markdown(
    md_content,
    extras=["tables", "fenced-code-blocks", "code-friendly", "break-on-newline", "header-ids"]
)

# Fix Table of Contents links to match actual header IDs
# The issue is that "&" in headers becomes "-" not "--" in the IDs
html_body = html_body.replace('href="#2-problem-statement--objectives"', 'href="#2-problem-statement-objectives"')
html_body = html_body.replace('href="#6-results--performance-analysis"', 'href="#6-results-performance-analysis"')
html_body = html_body.replace('href="#9-conclusions--future-work"', 'href="#9-conclusions-future-work"')

# Function to embed images as base64
def embed_image(img_path):
    try:
        with open(img_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
            ext = img_path.suffix.lower()
            mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
            return f'data:{mime_type};base64,{img_data}'
    except:
        return img_path

# Replace relative image paths with embedded base64
import re

def replace_img(match):
    img_src = match.group(1)
    if img_src.startswith('../results/'):
        img_path = RESULTS_DIR / img_src.replace('../results/', '')
        if img_path.exists():
            embedded = embed_image(img_path)
            return f'<img src="{embedded}" alt="{match.group(1)}"'
    return match.group(0)

html_body = re.sub(r'<img src="([^"]+)"', replace_img, html_body)

# Create full HTML with professional styling
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Amazon Product Categorization - NLP Project Report</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 15mm 20mm;
            }}
            
            body {{
                margin: 0;
                padding: 0;
            }}
            
            h1, h2, h3, h4 {{
                page-break-after: avoid;
            }}
            
            table, figure, img, pre {{
                page-break-inside: avoid;
            }}
            
            a[href^="http"]:after {{
                content: "";
            }}
        }}
        
        body {{
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20px;
            background: white;
        }}
        
        h1 {{
            font-size: 24pt;
            color: #1a1a1a;
            border-bottom: 3px solid #4a90e2;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
        }}
        
        h2 {{
            font-size: 18pt;
            color: #2c3e50;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        h3 {{
            font-size: 14pt;
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 12px;
        }}
        
        h4 {{
            font-size: 12pt;
            color: #555;
            margin-top: 15px;
            margin-bottom: 10px;
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
            font-family: 'Courier New', 'Consolas', monospace;
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
            margin: 15px 0;
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
        
        strong {{
            font-weight: bold;
            color: #1a1a1a;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 30px 0;
        }}
        
        .no-print {{
            display: block;
            background: #e8f4f8;
            padding: 15px;
            margin: 20px 0;
            border-left: 4px solid #4a90e2;
        }}
        
        @media print {{
            .no-print {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="no-print">
        <h3>📄 Print Instructions</h3>
        <p><strong>To save as PDF:</strong></p>
        <ol>
            <li>Press <strong>Ctrl + P</strong> (or Cmd + P on Mac)</li>
            <li>Select <strong>"Save as PDF"</strong> or <strong>"Microsoft Print to PDF"</strong> as the printer</li>
            <li>Click <strong>Save</strong></li>
            <li>Choose location and filename</li>
        </ol>
        <p>All images are embedded in this HTML file and will appear in the PDF.</p>
        <hr>
    </div>
    
    {html_body}
    
</body>
</html>"""

# Save HTML file
output_html = REPORT_DIR / "final_report.html"
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML file created: {output_html}")
print(f"File size: {output_html.stat().st_size / 1024:.2f} KB")
print("\nInstructions:")
print("1. Open final_report.html in your browser")
print("2. Press Ctrl+P (Print)")
print("3. Select 'Save as PDF' or 'Microsoft Print to PDF'")
print("4. Click Save")
print("\nAll images are embedded and will appear in the PDF!")
