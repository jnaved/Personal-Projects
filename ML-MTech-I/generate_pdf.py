import os
import sys
import argparse
import subprocess
import nbformat
import pygments
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import mistune

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def get_browser_path():
    if os.path.exists(CHROME_PATH):
        return CHROME_PATH
    elif os.path.exists(EDGE_PATH):
        return EDGE_PATH
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge was found on this system.")

def notebook_to_html(notebook_configs, doc_title="Machine Learning Assignment Report"):
    formatter = HtmlFormatter(style='friendly', nowrap=False)
    pygments_css = formatter.get_style_defs('.highlight')

    html_parts = []
    
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{doc_title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
        
        @page {{
            size: A4 portrait;
            margin: 12mm 12mm 15mm 12mm;
        }}
        
        * {{
            box-sizing: border-box;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #0f172a;
            line-height: 1.45;
            font-size: 11.5px;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        
        .report-header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            color: #ffffff;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .report-header h1 {{
            font-size: 19px;
            font-weight: 700;
            margin: 0 0 4px 0;
            color: #ffffff;
        }}
        
        .report-header .sub {{
            font-size: 12px;
            color: #bfdbfe;
            margin: 0;
        }}
        
        .page-break {{
            page-break-before: always;
        }}
        
        .question-card {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #2563eb;
            padding: 12px 16px;
            margin: 16px 0 10px 0;
            border-radius: 0 6px 6px 0;
            page-break-inside: avoid;
        }}
        
        .question-card h1, .question-card h2, .question-card h3, .question-card h4 {{
            margin-top: 0;
            margin-bottom: 6px;
            color: #1e3a8a;
            font-weight: 700;
        }}
        .question-card h1 {{ font-size: 15px; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; }}
        .question-card h2 {{ font-size: 14px; }}
        .question-card h3 {{ font-size: 13px; }}
        .question-card h4 {{ font-size: 12px; }}
        .question-card p {{ margin: 4px 0; font-size: 12px; color: #334155; }}
        .question-card ul, .question-card ol {{ margin: 4px 0; padding-left: 20px; font-size: 12px; color: #334155; }}
        .question-card li {{ margin-bottom: 2px; }}
        .question-card code {{ background: #e2e8f0; padding: 2px 4px; border-radius: 3px; font-family: 'Fira Code', monospace; font-size: 10.5px; }}
        
        .code-card {{
            margin-bottom: 8px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            overflow: hidden;
            page-break-inside: avoid;
            background: #ffffff;
        }}
        
        .card-header {{
            background-color: #f1f5f9;
            padding: 4px 10px;
            font-size: 10.5px;
            font-weight: 600;
            color: #334155;
            border-bottom: 1px solid #cbd5e1;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .badge-code {{
            background-color: #3b82f6;
            color: #ffffff;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 9.5px;
            font-weight: 600;
        }}
        
        .badge-output {{
            background-color: #059669;
            color: #ffffff;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 9.5px;
            font-weight: 600;
        }}
        
        .code-content {{
            background-color: #fcfcfc;
            padding: 8px 12px;
            font-family: 'Fira Code', Consolas, Monaco, monospace;
            font-size: 10.5px;
            overflow-x: auto;
        }}
        
        .code-content pre {{
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        
        .output-card {{
            margin-bottom: 14px;
            border: 1px solid #94a3b8;
            border-radius: 6px;
            overflow: hidden;
            page-break-inside: avoid;
            background-color: #ffffff;
        }}
        
        .output-header {{
            background-color: #e2e8f0;
            padding: 4px 10px;
            font-size: 10.5px;
            font-weight: 600;
            color: #1e293b;
            border-bottom: 1px solid #94a3b8;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .output-content {{
            padding: 10px;
            background-color: #ffffff;
        }}
        
        .stream-text {{
            background-color: #0f172a;
            color: #f8fafc;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: 'Fira Code', Consolas, Monaco, monospace;
            font-size: 10px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-break: break-all;
            margin: 0;
        }}
        
        .stderr-text {{
            background-color: #fff1f2;
            color: #9f1239;
            border-left: 3px solid #e11d48;
            padding: 6px 10px;
            font-family: 'Fira Code', Consolas, Monaco, monospace;
            font-size: 10px;
            white-space: pre-wrap;
            margin-top: 4px;
        }}
        
        .output-image {{
            display: block;
            max-width: 95%;
            height: auto;
            margin: 4px auto;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 4px 0;
        }}
        
        table.dataframe {{
            width: 100%;
            border-collapse: collapse;
            font-size: 10px;
            margin: 2px 0;
        }}
        
        table.dataframe th, table.dataframe td {{
            border: 1px solid #cbd5e1;
            padding: 4px 6px;
            text-align: left;
        }}
        
        table.dataframe th {{
            background-color: #f1f5f9;
            font-weight: 600;
            color: #0f172a;
        }}
        
        table.dataframe tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        
        {pygments_css}
    </style>
</head>
<body>
""")

    for nb_idx, config in enumerate(notebook_configs):
        nb_path = config["path"]
        title = config.get("title", f"Notebook {nb_idx + 1}")
        sub_info = config.get("sub", "")
        
        if nb_idx > 0:
            html_parts.append('<div class="page-break"></div>')
            
        html_parts.append(f"""
        <div class="report-header">
            <h1>{title}</h1>
            <div class="sub">Source File: {os.path.basename(nb_path)} {f' | {sub_info}' if sub_info else ''}</div>
        </div>
        """)
        
        nb = nbformat.read(nb_path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                md_html = mistune.html(cell.source)
                html_parts.append(f"""
                <div class="question-card">
                    {md_html}
                </div>
                """)
            elif cell.cell_type == 'code':
                source_code = cell.source.strip()
                if not source_code:
                    continue
                
                highlighted = pygments.highlight(source_code, PythonLexer(), formatter)
                exec_count = cell.get('execution_count')
                count_label = f"Cell [{exec_count}]" if exec_count else "Code"
                
                html_parts.append(f"""
                <div class="code-card">
                    <div class="card-header">
                        <span><strong>{count_label}</strong> &mdash; Implementation</span>
                        <span class="badge-code">Python Code</span>
                    </div>
                    <div class="code-content">
                        {highlighted}
                    </div>
                </div>
                """)
                
                raw_outputs = cell.get('outputs', [])
                # Filter out stderr warning streams
                outputs = []
                for out in raw_outputs:
                    if out.get('output_type') == 'stream' and out.get('name') == 'stderr':
                        text = out.get('text', '')
                        # Skip if it's a python warning
                        if any(w in text for w in ['Warning', 'warning', 'deprecated', 'Deprecated']):
                            continue
                    outputs.append(out)

                if outputs:
                    html_parts.append("""
                    <div class="output-card">
                        <div class="output-header">
                            <span><strong>Execution Output & Visualizations</strong></span>
                            <span class="badge-output">Output</span>
                        </div>
                        <div class="output-content">
                    """)
                    for out in outputs:
                        out_type = out.output_type
                        if out_type == 'stream':
                            text = out.get('text', '')
                            if out.get('name') == 'stderr':
                                html_parts.append(f'<div class="stderr-text">{text}</div>')
                            else:
                                html_parts.append(f'<pre class="stream-text">{text}</pre>')
                        elif out_type in ('execute_result', 'display_data'):
                            data = out.get('data', {})
                            if 'image/png' in data:
                                img_base64 = data['image/png']
                                html_parts.append(f'<img class="output-image" src="data:image/png;base64,{img_base64}" />')
                            elif 'text/html' in data:
                                html_content = "".join(data['text/html']) if isinstance(data['text/html'], list) else data['text/html']
                                html_parts.append(f'<div class="table-container">{html_content}</div>')
                            elif 'text/plain' in data:
                                text_plain = "".join(data['text/plain']) if isinstance(data['text/plain'], list) else data['text/plain']
                                html_parts.append(f'<pre class="stream-text">{text_plain}</pre>')
                        elif out_type == 'error':
                            traceback = "\n".join(out.get('traceback', []))
                            html_parts.append(f'<div class="stderr-text">{traceback}</div>')
                    html_parts.append('</div></div>')

    html_parts.append("""
</body>
</html>
""")
    return "\n".join(html_parts)

def convert_html_to_pdf(html_content, output_pdf_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    temp_html_path = output_pdf_path.replace(".pdf", "_temp.html")
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    browser = get_browser_path()
    abs_html = os.path.abspath(temp_html_path)
    abs_pdf = os.path.abspath(output_pdf_path)
    
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html}"
    ]
    
    subprocess.run(cmd, check=True)
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)
    print(f"Successfully created: {abs_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Convert Jupyter Notebooks to Assignment PDF reports.")
    parser.add_argument("notebooks", nargs="*", help="Paths to notebooks to convert (e.g. lab_1/q_1.ipynb lab_1/q_2.ipynb)")
    parser.add_argument("--combined", action="store_true", help="Combine all given notebooks into a single PDF")
    parser.add_argument("--output", "-o", help="Output PDF file path")
    args = parser.parse_args()

    if not args.notebooks:
        # Default workflow for Lab 1
        print("No specific notebook supplied. Generating all Lab 1 PDFs (individual + combined)...")
        
        # 1. Q1
        html_1 = notebook_to_html([{
            "path": "lab_1/q_1.ipynb",
            "title": "Lab 1 - Question 1: Exploratory Data Analysis on 20 Newsgroups Dataset",
            "sub": "M.Tech Machine Learning Lab Assignment"
        }], doc_title="Lab 1 - Question 1")
        convert_html_to_pdf(html_1, "lab_1/lab1_q1_report.pdf")
        
        # 2. Q2
        html_2 = notebook_to_html([{
            "path": "lab_1/q_2.ipynb",
            "title": "Lab 1 - Question 2: Used Cars Dataset Analysis & Preprocessing",
            "sub": "M.Tech Machine Learning Lab Assignment"
        }], doc_title="Lab 1 - Question 2")
        convert_html_to_pdf(html_2, "lab_1/lab1_q2_report.pdf")
        
        # 3. Combined
        html_comb = notebook_to_html([
            {
                "path": "lab_1/q_1.ipynb",
                "title": "Lab 1 - Question 1: Exploratory Data Analysis on 20 Newsgroups Dataset",
                "sub": "M.Tech Machine Learning Lab Assignment"
            },
            {
                "path": "lab_1/q_2.ipynb",
                "title": "Lab 1 - Question 2: Used Cars Dataset Analysis & Preprocessing",
                "sub": "M.Tech Machine Learning Lab Assignment"
            }
        ], doc_title="Machine Learning Lab 1 - Complete Assignment Report")
        convert_html_to_pdf(html_comb, "lab_1/lab1_complete_assignment.pdf")
    else:
        if args.combined:
            configs = [{"path": nb, "title": f"Assignment: {os.path.basename(nb)}"} for nb in args.notebooks]
            out_pdf = args.output or "combined_assignment_report.pdf"
            html = notebook_to_html(configs, doc_title="Combined Assignment Report")
            convert_html_to_pdf(html, out_pdf)
        else:
            for nb in args.notebooks:
                out_pdf = args.output or nb.replace(".ipynb", ".pdf")
                html = notebook_to_html([{"path": nb, "title": f"Assignment: {os.path.basename(nb)}"}], doc_title=os.path.basename(nb))
                convert_html_to_pdf(html, out_pdf)

if __name__ == "__main__":
    main()
