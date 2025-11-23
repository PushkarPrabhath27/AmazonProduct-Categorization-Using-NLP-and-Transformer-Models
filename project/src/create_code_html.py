"""
Convert PROJECT_CODE_AND_LOGIC.md to HTML for easy PDF printing
"""
import markdown2
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DOC = PROJECT_ROOT / "PROJECT_CODE_AND_LOGIC.md"

# Read the markdown file
with open(CODE_DOC, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_body = markdown2.markdown(
    md_content,
    extras=["fenced-code-blocks", "code-friendly", "break-on-newline", "header-ids"]
)

# Create full HTML with professional styling (same as report)
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Code and Logic Documentation</title>
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
            
            pre {{
                page-break-inside: avoid;
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
        
        code {{
            font-family: 'Courier New', 'Consolas', monospace;
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
            <li>Press <strong>Ctrl + P</strong></li>
            <li>Select <strong>"Save as PDF"</strong></li>
            <li>Click <strong>Save</strong></li>
        </ol>
        <hr>
    </div>
    
    {html_body}
    
</body>
</html>"""

# Save HTML file
output_html = PROJECT_ROOT / "PROJECT_CODE_AND_LOGIC.html"
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML file created: {output_html}")
