#convert_md_to_pdf.py:
#!/usr/bin/env python
"""Convert markdown to PDF using markdown2 and weasyprint"""

import markdown2
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def markdown_to_pdf(md_file, pdf_file):
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(md_content, extras=[
        'fenced-code-blocks',
        'tables',
        'code-friendly',
        'break-on-newline'
    ])
    
    # Create styled HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Arial', 'Helvetica', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 8px;
                margin-top: 25px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }}
            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-left: 4px solid #3498db;
                padding: 15px;
                overflow-x: auto;
                border-radius: 4px;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                padding-left: 20px;
                margin-left: 0;
                color: #555;
                font-style: italic;
            }}
            .page-break {{
                page-break-after: always;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Create PDF
    print(f"Converting {md_file} to {pdf_file}...")
    font_config = FontConfiguration()
    HTML(string=full_html).write_pdf(
        pdf_file,
        font_config=font_config
    )
    print(f"✅ PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    md_file = r"C:\Users\CSE-312-01\.gemini\antigravity\brain\1b841313-0930-42c8-9406-fe54c0fbadaa\python_script_explanation.md"
    pdf_file = r"C:\Users\CSE-312-01\Downloads\Capstone Project 1992\DeepTrace_Python_Script_Explanation.pdf"
    
    markdown_to_pdf(md_file, pdf_file)
