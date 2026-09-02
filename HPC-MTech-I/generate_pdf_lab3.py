import os
import subprocess
import shutil
import base64
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def get_browser():
    if os.path.exists(EDGE_PATH):
        return EDGE_PATH
    if os.path.exists(CHROME_PATH):
        return CHROME_PATH
    raise FileNotFoundError("Browser not found")

def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def format_code_lines(code_text):
    lines = code_text.strip().splitlines()
    html_lines = []
    for i, line in enumerate(lines, start=1):
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Simple C syntax highlighting cues
        for kw in ["#include", "#define", "void", "int", "float", "double", "return", "if", "for", "const", "sizeof"]:
            escaped = escaped.replace(f"{kw} ", f"<span class='c-kw'>{kw}</span> ")
        
        if "#pragma omp" in escaped:
            escaped = escaped.replace("#pragma omp", "<span class='c-omp'>#pragma omp</span>")
        
        if escaped.strip().startswith("/*") or escaped.strip().startswith("*") or escaped.strip().startswith("//"):
            escaped = f"<span class='c-comment'>{escaped}</span>"
            
        html_lines.append(f"<div class='code-row'><span class='line-no'>{i}</span><span class='line-code'>{escaped}</span></div>")
    return "\n".join(html_lines)

def build_html_report(base_dir):
    lab3_dir = base_dir / "lab_3"
    images_dir = lab3_dir / "images"
    
    q1_code = (lab3_dir / "q1.c").read_text(encoding="utf-8")
    q1_img_b64 = get_image_base64(images_dir / "q1.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HPC Lab Assignment 3 - Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Tinos:ital,wght@0,400;0,700;1,400;1,700&family=JetBrains+Mono:wght@400;500;700&display=swap');

@page {{
    size: A4;
    margin: 18mm 18mm 18mm 18mm;
    @top-center {{
        content: "";
    }}
}}

* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}

body {{
    font-family: 'Tinos', 'Times New Roman', Times, serif;
    font-size: 11.5pt;
    line-height: 1.42;
    color: #111111;
    background: #ffffff;
}}

.header-rule {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 0.8pt solid #000000;
    padding-bottom: 4px;
    margin-bottom: 14px;
    font-size: 11pt;
    font-weight: normal;
}}

.header-rule u {{
    text-underline-offset: 3px;
}}

.center-title {{
    text-align: center;
    margin-bottom: 14px;
    line-height: 1.35;
}}

.center-title .dept {{
    font-size: 12.5pt;
    font-weight: bold;
}}

.center-title .deg {{
    font-size: 11.5pt;
    font-weight: bold;
}}

.center-title .sub {{
    font-size: 11.5pt;
    font-weight: bold;
}}

.center-title .lab {{
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 2px;
}}

.section-h {{
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 5px;
}}

.prob-title {{
    font-size: 11.5pt;
    font-weight: bold;
    margin-bottom: 4px;
}}

.prob-desc {{
    font-size: 11pt;
    text-align: justify;
    margin-bottom: 10px;
}}

/* Code Box */
.code-container {{
    border: 0.8pt solid #333333;
    background: #fafafa;
    border-radius: 2px;
    padding: 5px 8px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 8pt;
    line-height: 1.26;
    margin-bottom: 10px;
}}

.code-row {{
    display: flex;
}}

.line-no {{
    width: 28px;
    color: #888888;
    user-select: none;
    text-align: right;
    padding-right: 8px;
    font-size: 7.5pt;
}}

.line-code {{
    flex: 1;
    white-space: pre-wrap;
    word-break: break-all;
}}

.c-kw {{ font-weight: bold; color: #0033aa; }}
.c-omp {{ font-weight: bold; color: #a00060; }}
.c-comment {{ color: #007700; font-style: italic; }}

/* Image & Output */
.figure-box {{
    text-align: center;
    margin: 10px 0 12px 0;
}}

.figure-box img {{
    max-width: 96%;
    height: auto;
    border-radius: 4px;
    border: 0.5pt solid #cccccc;
}}

.figure-caption {{
    font-size: 10pt;
    font-weight: normal;
    margin-top: 4px;
    text-align: center;
    color: #222222;
}}

/* Analysis */
.analysis-body {{
    text-align: justify;
    font-size: 11pt;
    line-height: 1.42;
}}

.analysis-p {{
    margin-bottom: 10px;
}}

.analysis-p strong {{
    font-weight: bold;
}}

.analysis-p ul, .analysis-p ol {{
    margin-left: 22px;
    margin-top: 4px;
    margin-bottom: 6px;
}}

.math-eq {{
    text-align: center;
    margin: 5px 0;
    font-style: italic;
    font-size: 11pt;
}}

/* Table */
table.academic-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 14px 0;
    font-size: 10pt;
}}

table.academic-table th, table.academic-table td {{
    border-top: 0.8pt solid #000000;
    border-bottom: 0.8pt solid #000000;
    padding: 6px 8px;
    text-align: left;
}}

table.academic-table th {{
    font-weight: bold;
    background: #fdfdfd;
}}

.page-break {{
    page-break-before: always;
}}
</style>
</head>
<body>

<div class="header-rule">
    <div>Name: <u>Shaik Jnaved</u></div>
    <div>Enrollment No.: <u>P26DS001</u></div>
</div>

<div class="center-title">
    <div class="dept">Computer Science and Engineering Department, SVNIT, Surat</div>
    <div class="deg">M.Tech. I -- Semester I</div>
    <div class="sub">High Performance Computing</div>
    <div class="lab">Lab Assignment 3</div>
</div>

<div class="section-h">Problem Statement</div>
<div class="prob-title">Title: Parallel 3D Linear Regression using OpenMP and Sequential Performance Comparison</div>
<div class="prob-desc">
    Implement a parallel version of multiple linear regression using OpenMP on a 3-dimensional dataspace (X<sub>1</sub>, X<sub>2</sub> &isin; &reals; features, and Y &isin; &reals; label). Apply data-parallel gradient descent to optimize parameters w<sub>1</sub>, w<sub>2</sub>, b minimizing the Mean Squared Error (MSE) loss function over N = 2,000,000 synthetic datapoints and 100 epochs. Evaluate and compare execution runtime, speedup, and parallel efficiency across varying OpenMP thread configurations (1, 2, 4, 8 threads) against the baseline sequential algorithm.
</div>

<div class="section-h">Source Code:</div>
<div class="code-container">
{format_code_lines(q1_code)}
</div>

<div class="page-break"></div>

<div class="header-rule">
    <div>Name: <u>Shaik Jnaved</u></div>
    <div>Enrollment No.: <u>P26DS001</u></div>
</div>

<div class="section-h">Output:</div>
<div class="figure-box">
    <img src="data:image/png;base64,{q1_img_b64}" alt="Parallel Linear Regression Output" />
    <div class="figure-caption">Figure 1: Program Execution and Scalability Benchmark Output for Problem 1 (N = 2,000,000, Epochs = 100)</div>
</div>

<div class="section-h">Analysis:</div>
<div class="analysis-body">
    <div class="analysis-p">
        <strong>1. OpenMP Parallel Data Decomposition &amp; Reduction:</strong>
        <ul>
            <li><strong>Work-Sharing Directive:</strong> <code>#pragma omp parallel for reduction(+:dw1, dw2, db) num_threads(T) schedule(static)</code> parallelizes the inner loop computing error contributions across N = 2,000,000 points.</li>
            <li><strong>Static Scheduling &amp; Cache Locality:</strong> With <code>schedule(static)</code>, iterations are partitioned into contiguous blocks of size &lfloor;N / T&rfloor;. Because arrays <code>x1</code>, <code>x2</code>, and <code>y</code> are stored in contiguous 1D memory buffers, sequential thread accesses maximize L1/L2 hardware prefetching and eliminate runtime task scheduling overhead.</li>
            <li><strong>Multi-Variable Reduction:</strong> The <code>reduction(+:dw1, dw2, db)</code> clause creates private copies of <code>dw1</code>, <code>dw2</code>, and <code>db</code> for each thread, initialized to 0.0f. Each thread accumulates partial sums locally in CPU registers/private cache lines. At loop termination, OpenMP combines all private copies into the master variables using a lock-free tree reduction, avoiding race conditions and eliminating cache-line false sharing.</li>
        </ul>
    </div>

    <div class="analysis-p">
        <strong>2. Scalability, Speedup, and Efficiency Analysis:</strong><br>
        Parallel performance metrics are evaluated using:
        <div class="math-eq">Speedup S<sub>p</sub> = T<sub>1</sub> / T<sub>p</sub>, &emsp; Parallel Efficiency E<sub>p</sub> = (S<sub>p</sub> / p) &times; 100%</div>
        <ul>
            <li><strong>2 Threads:</strong> Execution time decreases from 0.161999 s to 0.084922 s, achieving a speedup of <strong>1.91&times;</strong> and near-ideal parallel efficiency of <strong>95.4%</strong>.</li>
            <li><strong>4 Threads:</strong> Execution time drops to 0.049519 s, yielding a speedup of <strong>3.27&times;</strong> and an efficiency of <strong>81.8%</strong>.</li>
            <li><strong>8 Threads:</strong> Execution time reaches 0.048290 s (<strong>3.35&times;</strong> speedup, <strong>41.9%</strong> efficiency).</li>
        </ul>
    </div>

    <div class="analysis-p">
        <strong>3. Architectural Bottlenecks and Saturation (Memory Bandwidth &amp; Amdahl's Law):</strong>
        <ul>
            <li><strong>Memory-Bound Nature:</strong> Each epoch streams 3 &times; 2,000,000 &times; 4 bytes = 24 MB of data across the memory bus. Across 100 epochs, 2.4 GB is streamed. At 8 threads, the arithmetic intensity is low (&approx; 5 FLOPs per 12 bytes read = 0.42 FLOP/byte), leading to memory bus bandwidth saturation where ALU cores stall waiting for RAM/LLC data prefetching.</li>
            <li><strong>Inter-Epoch Barrier Synchronization:</strong> At the end of each epoch, all threads synchronize at the reduction barrier while the master thread executes the scalar parameter update. This sequential fraction imposes an asymptotic speedup ceiling as dictated by Amdahl's Law.</li>
        </ul>
    </div>

    <table class="academic-table">
        <thead>
            <tr>
                <th>Configuration</th>
                <th>Threads (p)</th>
                <th>Execution Time (T<sub>p</sub>)</th>
                <th>Speedup (S<sub>p</sub>)</th>
                <th>Efficiency (E<sub>p</sub>)</th>
                <th>Final MSE Loss</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Sequential Baseline</td>
                <td>1</td>
                <td>0.161999 s</td>
                <td>1.00&times;</td>
                <td>100.0%</td>
                <td>0.040441</td>
            </tr>
            <tr>
                <td>OpenMP Parallel</td>
                <td>2</td>
                <td>0.084922 s</td>
                <td>1.91&times;</td>
                <td>95.4%</td>
                <td>0.040423</td>
            </tr>
            <tr>
                <td>OpenMP Parallel</td>
                <td>4</td>
                <td>0.049519 s</td>
                <td>3.27&times;</td>
                <td>81.8%</td>
                <td>0.040423</td>
            </tr>
            <tr>
                <td>OpenMP Parallel</td>
                <td>8</td>
                <td>0.048290 s</td>
                <td>3.35&times;</td>
                <td>41.9%</td>
                <td>0.040423</td>
            </tr>
        </tbody>
    </table>
</div>

</body>
</html>"""
    return html

def convert_html_to_pdf(html_content, output_pdf_path):
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    temp_html = output_pdf_path.with_suffix(".temp.html")
    temp_html.write_text(html_content, encoding="utf-8")
    
    browser = get_browser()
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf_path.resolve()}",
        f"file:///{temp_html.resolve()}"
    ]
    subprocess.run(cmd, check=True)
    if temp_html.exists():
        temp_html.unlink()
    print(f"[Generated PDF] {output_pdf_path} ({os.path.getsize(output_pdf_path)} bytes)")

def main():
    base_dir = Path(__file__).parent
    html = build_html_report(base_dir)
    
    pdf_lab3 = base_dir / "lab_3" / "lab_assignment.pdf"
    convert_html_to_pdf(html, pdf_lab3)
    print(f"[SUCCESS] Lab 3 Assignment PDF generated successfully at:\n  - {pdf_lab3}")

if __name__ == "__main__":
    main()
