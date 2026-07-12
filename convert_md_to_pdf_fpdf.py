#convert_md_to_pdf_fpdf.py:
import os
import re
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, 'DeepTrace Documentation', border=False, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def markdown_to_pdf_fpdf(md_file, pdf_file):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pre-process content
    # 1. Strip emojis and non-latin-1
    content = content.encode('latin-1', 'ignore').decode('latin-1')
    
    # 2. Split into blocks
    lines = content.split('\n')

    current_font_size = 11
    pdf.set_font("helvetica", size=current_font_size)
    
    # Available width for A4 (210mm) - margins (20mm) = 190mm
    W = 180 
    
    in_code_block = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
            
        if in_code_block:
            pdf.set_font("courier", size=9)
            pdf.multi_cell(W, 5, line)
            pdf.set_font("helvetica", size=current_font_size)
            continue

        # Headers
        if line.startswith("# "):
            pdf.ln(5)
            pdf.set_font("helvetica", 'B', 18)
            pdf.multi_cell(W, 10, line[2:])
            pdf.set_font("helvetica", size=current_font_size)
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(3)
            pdf.set_font("helvetica", 'B', 16)
            pdf.multi_cell(W, 10, line[3:])
            pdf.set_font("helvetica", size=current_font_size)
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.ln(2)
            pdf.set_font("helvetica", 'B', 14)
            pdf.multi_cell(W, 10, line[4:])
            pdf.set_font("helvetica", size=current_font_size)
        elif line.startswith("- "):
            pdf.multi_cell(W, 7, f"  - {line[2:]}")
        elif "|" in line:
            # Skip tables or treat as plain text
            pdf.set_font("helvetica", 'I', 9)
            pdf.multi_cell(W, 5, line)
            pdf.set_font("helvetica", size=current_font_size)
        else:
            # Bold/Italic cleanup
            line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            line = re.sub(r'\*(.*?)\*', r'\1', line)
            line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', line)
            
            if line:
                pdf.multi_cell(W, 7, line)
            else:
                pdf.ln(3)

    print(f"Saving PDF to {pdf_file}...")
    pdf.output(pdf_file)
    print("Done!")

if __name__ == "__main__":
    md_file = r"C:\Users\CSE-312-01\.gemini\antigravity\brain\1b841313-0930-42c8-9406-fe54c0fbadaa\python_script_explanation.md"
    pdf_file = r"C:\Users\CSE-312-01\Downloads\Capstone Project 1992\DeepTrace_Python_Script_Explanation.pdf"
    
    markdown_to_pdf_fpdf(md_file, pdf_file)
