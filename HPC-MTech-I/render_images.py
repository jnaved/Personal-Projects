import os
import subprocess
import shutil
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def get_browser():
    if os.path.exists(EDGE_PATH):
        return EDGE_PATH
    if os.path.exists(CHROME_PATH):
        return CHROME_PATH
    raise FileNotFoundError("Browser not found")

def render_terminal_html(title, content_lines, width=780):
    formatted_lines = []
    for line in content_lines:
        line_clean = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if "Expected Value" in line_clean or "Actual Value" in line_clean or "Time taken" in line_clean:
            formatted_lines.append(f"<span class='hl-accent'>{line_clean}</span>")
        elif "Thread ID" in line_clean:
            formatted_lines.append(f"<span class='hl-thread'>{line_clean}</span>")
        elif "Number of" in line_clean:
            formatted_lines.append(f"<span class='hl-info'>{line_clean}</span>")
        else:
            formatted_lines.append(line_clean)
    
    body_text = "\n".join(formatted_lines)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: transparent;
    padding: 24px;
    font-family: 'Consolas', 'Courier New', monospace;
    display: inline-block;
}}
.terminal-window {{
    width: {width}px;
    background: #1a1b26;
    border-radius: 10px;
    border: 1px solid #414868;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45);
    overflow: hidden;
}}
.terminal-header {{
    background: #24283b;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #414868;
}}
.mac-dots {{
    display: flex;
    gap: 8px;
}}
.mac-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
}}
.dot-red {{ background: #ff5f56; border: 1px solid #e0443e; }}
.dot-yellow {{ background: #ffbd2e; border: 1px solid #dea123; }}
.dot-green {{ background: #27c93f; border: 1px solid #1aab29; }}
.window-title {{
    flex: 1;
    text-align: center;
    color: #7aa2f7;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-right: 48px;
}}
.terminal-body {{
    padding: 20px 22px;
    color: #c0caf5;
    font-size: 14.5px;
    line-height: 1.6;
    white-space: pre-wrap;
}}
.hl-accent {{ color: #7dcfff; font-weight: bold; }}
.hl-thread {{ color: #bb9af7; }}
.hl-info {{ color: #e0af68; }}
</style>
</head>
<body>
<div class="terminal-window">
    <div class="terminal-header">
        <div class="mac-dots">
            <span class="mac-dot dot-red"></span>
            <span class="mac-dot dot-yellow"></span>
            <span class="mac-dot dot-green"></span>
        </div>
        <div class="window-title">{title}</div>
    </div>
    <div class="terminal-body">{body_text}</div>
</div>
</body>
</html>"""
    return html

def render_diagram_html():
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #ffffff;
    padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    display: inline-block;
    color: #1e293b;
}
.card-container {
    width: 900px;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    background: #f8fafc;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    overflow: hidden;
}
.header-bar {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: #ffffff;
    padding: 16px 20px;
    text-align: center;
}
.header-bar h2 {
    font-size: 17.5px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.header-bar p {
    font-size: 12.5px;
    color: #94a3b8;
    margin-top: 4px;
}
.grid-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
    padding: 22px;
}
.method-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}
.method-box.coarse {
    border-top: 4px solid #10b981;
}
.method-box.fine {
    border-top: 4px solid #3b82f6;
}
.method-title {
    font-size: 15px;
    font-weight: bold;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.badge-coarse {
    background: #d1fae5;
    color: #065f46;
    font-size: 11.5px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 600;
}
.badge-fine {
    background: #dbeafe;
    color: #1e40af;
    font-size: 11.5px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 600;
}
.matrix-visual {
    margin: 12px 0 16px 0;
    background: #f1f5f9;
    border-radius: 8px;
    padding: 12px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
}
.matrix-row {
    display: flex;
    gap: 4px;
    margin-bottom: 5px;
}
.matrix-row:last-child {
    margin-bottom: 0;
}
.chunk-c0 { background: #6ee7b7; color: #064e3b; flex: 1; text-align: center; padding: 5px 0; border-radius: 4px; font-weight: bold; }
.chunk-c1 { background: #a7f3d0; color: #064e3b; flex: 1; text-align: center; padding: 5px 0; border-radius: 4px; font-weight: bold; }
.chunk-c2 { background: #34d399; color: #064e3b; flex: 1; text-align: center; padding: 5px 0; border-radius: 4px; font-weight: bold; }
.chunk-c3 { background: #059669; color: #ffffff; flex: 1; text-align: center; padding: 5px 0; border-radius: 4px; font-weight: bold; }

.elem-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
}
.cell-f { background: #93c5fd; color: #1e3a8a; text-align: center; padding: 5px 0; border-radius: 4px; font-weight: bold; font-size: 11px; }

.feature-list {
    list-style: none;
    font-size: 12.5px;
    line-height: 1.6;
    color: #334155;
    margin-top: 10px;
}
.feature-list li {
    margin-bottom: 7px;
    display: flex;
    align-items: baseline;
    gap: 6px;
}
.bullet {
    color: #64748b;
    font-weight: bold;
}
.feat-label {
    font-weight: 600;
    color: #0f172a;
    min-width: 90px;
}
.feat-val {
    flex: 1;
}
code {
    background: #f1f5f9;
    padding: 2px 5px;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 11.5px;
    color: #0f172a;
}
.timing-highlight {
    margin-top: 14px;
    padding: 10px 14px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 13.5px;
    text-align: center;
}
.timing-highlight.coarse-time {
    background: #ecfdf5;
    color: #047857;
    border: 1.5px dashed #10b981;
}
.timing-highlight.fine-time {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1.5px dashed #3b82f6;
}
.footer-summary {
    background: #f1f5f9;
    padding: 14px 22px;
    border-top: 1px solid #e2e8f0;
    font-size: 12.5px;
    color: #475569;
    line-height: 1.5;
}
</style>
</head>
<body>
<div class="card-container">
    <div class="header-bar">
        <h2>Data Decomposition Granularity: Coarse-Grained vs. Fine-Grained</h2>
        <p>Comparative Architecture & Runtime Performance for N = 1500 on 8 OpenMP Threads</p>
    </div>
    <div class="grid-content">
        <div class="method-box coarse">
            <div class="method-title">
                <span>Coarse-Grained (1D Outer Loop)</span>
                <span class="badge-coarse">Faster (~1.8x)</span>
            </div>
            <div class="matrix-visual">
                <div class="matrix-row"><div class="chunk-c0">Thread 0 (Rows 0 .. N/4 - 1)</div></div>
                <div class="matrix-row"><div class="chunk-c1">Thread 1 (Rows N/4 .. N/2 - 1)</div></div>
                <div class="matrix-row"><div class="chunk-c2">Thread 2 (Rows N/2 .. 3N/4 - 1)</div></div>
                <div class="matrix-row"><div class="chunk-c3">Thread 3 (Rows 3N/4 .. N - 1)</div></div>
            </div>
            <ul class="feature-list">
                <li><span class="bullet">&bull;</span> <span class="feat-label">Directive:</span> <span class="feat-val"><code>#pragma omp parallel for private(j,k)</code></span></li>
                <li><span class="bullet">&bull;</span> <span class="feat-label">Task Size:</span> <span class="feat-val">N large contiguous row blocks assigned to threads.</span></li>
                <li><span class="bullet">&bull;</span> <span class="feat-label">Locality:</span> <span class="feat-val">Excellent contiguous row memory reuse for A & C.</span></li>
                <li><span class="bullet">&bull;</span> <span class="feat-label">Overhead:</span> <span class="feat-val">Minimal (only N iterations partitioned).</span></li>
            </ul>
            <div class="timing-highlight coarse-time">
                Execution Time: 0.345000 s
            </div>
        </div>

        <div class="method-box fine">
            <div class="method-title">
                <span>Fine-Grained (2D Collapsed)</span>
                <span class="badge-fine">Higher Overhead</span>
            </div>
            <div class="matrix-visual">
                <div class="elem-grid">
                    <div class="cell-f">T0(0,0)</div><div class="cell-f">T1(0,1)</div><div class="cell-f">T2(0,2)</div><div class="cell-f">T3(0,3)</div>
                    <div class="cell-f">T4(1,0)</div><div class="cell-f">T5(1,1)</div><div class="cell-f">T6(1,2)</div><div class="cell-f">T7(1,3)</div>
                    <div class="cell-f">T0(2,0)</div><div class="cell-f">T1(2,1)</div><div class="cell-f">T2(2,2)</div><div class="cell-f">T3(2,3)</div>
                </div>
            </div>
            <ul class="feature-list">
                <li><span class="bullet">&bull;</span> <span class="feat-label">Directive:</span> <span class="feat-val"><code>#pragma omp parallel for collapse(2)</code></span></li>
                <li><span class="bullet">&bull;</span> <span class="feat-label">Task Size:</span> <span class="feat-val">N&sup2; tiny element-level tasks (2,250,000 tasks).</span></li>
                <li><span class="bullet">&bull;</span> <span class="feat-label">Locality:</span> <span class="feat-val">Moderate (frequent thread context switches).</span></li>
                <li><span class="bullet">&bull;</span> <span class="feat-label">Overhead:</span> <span class="feat-val">Significant index calculation & dispatch load.</span></li>
            </ul>
            <div class="timing-highlight fine-time">
                Execution Time: 0.629000 s
            </div>
        </div>
    </div>
    <div class="footer-summary">
        <strong>Key Architectural Takeaway:</strong> For memory-bound and compute-dense matrix multiplication, 1D outer-loop coarse-grained decomposition minimizes OpenMP runtime dispatch overhead and maximizes L1/L2 data cache hit ratios along contiguous C row-major memory.
    </div>
</div>
</body>
</html>"""
    return html

def html_to_png(html_content, output_png_path, width=860, height=520):
    browser = get_browser()
    temp_html = output_png_path.with_suffix(".temp.html")
    temp_html.write_text(html_content, encoding="utf-8")
    
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        f"--screenshot={output_png_path.resolve()}",
        f"file:///{temp_html.resolve()}"
    ]
    subprocess.run(cmd, check=True)
    if temp_html.exists():
        temp_html.unlink()
    print(f"[Generated Image] {output_png_path} ({os.path.getsize(output_png_path)} bytes)")

def read_file_safe(path):
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc).strip().splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").strip().splitlines()

def main():
    base_dir = Path(__file__).parent
    lab1_images = base_dir / "lab_1" / "images"
    overleaf_images = base_dir / "overleaf" / "lab_1" / "images"
    
    lab1_images.mkdir(parents=True, exist_ok=True)
    overleaf_images.mkdir(parents=True, exist_ok=True)
    
    # Q1
    q1_text = read_file_safe(base_dir / "lab_1" / "q1_stdout.txt")
    q1_html = render_terminal_html("Terminal Output - Problem 1: Race Condition Demonstration", q1_text)
    q1_img = lab1_images / "q1_output.png"
    html_to_png(q1_html, q1_img, width=820, height=360)
    
    # Q2
    q2_text = read_file_safe(base_dir / "lab_1" / "q2_stdout.txt")
    q2_html = render_terminal_html("Terminal Output - Problem 2: Critical Section Synchronization", q2_text)
    q2_img = lab1_images / "q2_output.png"
    html_to_png(q2_html, q2_img, width=820, height=360)
    
    # Q3
    q3_text = read_file_safe(base_dir / "lab_1" / "q3_stdout.txt")
    q3_html = render_terminal_html("Terminal Output - Problem 3: Parallel Matrix Multiplication Benchmark", q3_text)
    q3_img = lab1_images / "q3_output.png"
    html_to_png(q3_html, q3_img, width=820, height=260)
    
    # Q3 Diagram
    q3_diag_html = render_diagram_html()
    q3_diag_img = lab1_images / "q3_decomposition_comparison.png"
    html_to_png(q3_diag_html, q3_diag_img, width=960, height=560)
    
    # Copy all images to overleaf directory
    for img in [q1_img, q2_img, q3_img, q3_diag_img]:
        shutil.copy(img, overleaf_images / img.name)
    
    print("[SUCCESS] All images rendered and mirrored to overleaf directory!")

if __name__ == "__main__":
    main()
