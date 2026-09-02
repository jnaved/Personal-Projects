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
        
        # Simple C syntax cues
        for kw in ["#include", "#define", "void", "int", "long long", "double", "return", "if", "for", "bool", "true", "false", "char"]:
            escaped = escaped.replace(f"{kw} ", f"<span class='c-kw'>{kw}</span> ")
        
        if "#pragma omp" in escaped:
            escaped = escaped.replace("#pragma omp", "<span class='c-omp'>#pragma omp</span>")
        
        if escaped.strip().startswith("/*") or escaped.strip().startswith("*") or escaped.strip().startswith("//"):
            escaped = f"<span class='c-comment'>{escaped}</span>"
            
        html_lines.append(f"<div class='code-row'><span class='line-no'>{i}</span><span class='line-code'>{escaped}</span></div>")
    return "\n".join(html_lines)

def build_html_report(base_dir):
    lab1_dir = base_dir / "lab_1"
    images_dir = lab1_dir / "images"
    
    q1_code = (lab1_dir / "q1.c").read_text(encoding="utf-8")
    q2_code = (lab1_dir / "q2.c").read_text(encoding="utf-8")
    q3_code = (lab1_dir / "q3.c").read_text(encoding="utf-8")
    
    q1_img_b64 = get_image_base64(images_dir / "q1_output.png")
    q2_img_b64 = get_image_base64(images_dir / "q2_output.png")
    q3_img_b64 = get_image_base64(images_dir / "q3_output.png")
    q3_diag_b64 = get_image_base64(images_dir / "q3_decomposition_comparison.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HPC Lab Assignment 1 - Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Tinos:ital,wght@0,400;0,700;1,400;1,700&family=JetBrains+Mono:wght@400;500;700&display=swap');

@page {{
    size: A4;
    margin: 22mm 18mm 20mm 18mm;
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
    font-size: 12pt;
    line-height: 1.45;
    color: #111111;
    background: #ffffff;
}}

.header-rule {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 0.8pt solid #000000;
    padding-bottom: 4px;
    margin-bottom: 18px;
    font-size: 11pt;
    font-weight: normal;
}}

.header-rule u {{
    text-underline-offset: 3px;
}}

.center-title {{
    text-align: center;
    margin-bottom: 18px;
    line-height: 1.4;
}}

.center-title .dept {{
    font-size: 12.5pt;
    font-weight: bold;
}}

.center-title .deg {{
    font-size: 12pt;
    font-weight: bold;
}}

.center-title .sub {{
    font-size: 12pt;
    font-weight: bold;
}}

.center-title .lab {{
    font-size: 12pt;
    font-weight: bold;
    margin-top: 2px;
}}

.section-h {{
    font-size: 12pt;
    font-weight: bold;
    margin-top: 14px;
    margin-bottom: 6px;
}}

.prob-title {{
    font-size: 12pt;
    font-weight: bold;
    margin-bottom: 4px;
}}

.prob-desc {{
    font-size: 12pt;
    text-align: justify;
    margin-bottom: 12px;
}}

.prob-desc ol, .prob-desc ul {{
    margin-left: 24px;
    margin-top: 4px;
    margin-bottom: 6px;
}}

/* Code Box */
.code-container {{
    border: 0.8pt solid #333333;
    background: #fafafa;
    border-radius: 2px;
    padding: 6px 10px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 9pt;
    line-height: 1.35;
    margin-bottom: 16px;
}}

.code-row {{
    display: flex;
}}

.line-no {{
    width: 28px;
    color: #888888;
    user-select: none;
    text-align: right;
    padding-right: 10px;
    font-size: 8.5pt;
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
    margin: 14px 0 16px 0;
}}

.figure-box img {{
    max-width: 95%;
    height: auto;
    border-radius: 4px;
    border: 0.5pt solid #cccccc;
}}

.figure-caption {{
    font-size: 10.5pt;
    font-weight: normal;
    margin-top: 6px;
    text-align: center;
    color: #222222;
}}

/* Analysis */
.analysis-body {{
    text-align: justify;
    font-size: 12pt;
    line-height: 1.45;
}}

.analysis-p {{
    margin-bottom: 10px;
}}

.analysis-p strong {{
    font-weight: bold;
}}

.analysis-p ul, .analysis-p ol {{
    margin-left: 24px;
    margin-top: 4px;
    margin-bottom: 8px;
}}

.math-eq {{
    text-align: center;
    margin: 8px 0;
    font-style: italic;
    font-size: 12pt;
}}

/* Table */
table.academic-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 16px 0;
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
    background: #f4f4f4;
    border-top: 1.5pt solid #000000;
    border-bottom: 1.2pt solid #000000;
}}

table.academic-table tr:last-child td {{
    border-bottom: 1.5pt solid #000000;
}}

.page-break {{
    page-break-before: always;
    break-before: page;
}}
</style>
</head>
<body>

<!-- ======================================================================= -->
<!-- PROBLEM 1 -->
<!-- ======================================================================= -->

<div class="header-rule">
    <div>Name: <u>Shaik Jnaved</u></div>
    <div>Enrollment No.: <u>P26DS001</u></div>
</div>

<div class="center-title">
    <div class="dept">Computer Science and Engineering Department, SVNIT, Surat</div>
    <div class="deg">M.Tech. I -- Semester I</div>
    <div class="sub">High Performance Computing</div>
    <div class="lab">Lab Assignment 1 -- Problem 1</div>
</div>

<div class="section-h">Problem Statement</div>

<div class="prob-title">Title: Multi-Threaded Shared Resource Contention and Race Condition Demonstration</div>
<div class="prob-desc">
Demonstrate a race condition between multiple competing threads accessing and modifying a shared resource in a shared-memory parallel architecture using OpenMP. Configure an execution environment with 4 concurrent threads, where each thread independently executes 50,000 increment iterations on a common global counter variable without any synchronization constructs. Measure the final aggregate value of the counter, compare it against the expected deterministic value (4 &times; 50,000 = 200,000), and analyze the underlying architectural and memory-consistency causes of data loss.
</div>

<div class="section-h">Code:</div>
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
    <img src="data:image/png;base64,{q1_img_b64}" alt="Q1 Output Screenshot" />
    <div class="figure-caption">Figure 1: Execution Output for Problem 1: Race Condition in Unsynchronized OpenMP Threads</div>
</div>

<div class="section-h">Analysis:</div>
<div class="analysis-body">
    <div class="analysis-p">
        <strong>1. Shared-Memory Parallel Model and Thread Contention:</strong><br>
        In the OpenMP shared-memory programming model, all threads spawned within the <code>#pragma omp parallel</code> directive share a unified virtual address space. The variable <code>shared_counter</code> is defined in the outer scope of the parallel region and is treated as a <strong>shared</strong> variable across all worker threads. When multiple threads concurrently execute <code>shared_counter += 1</code> without explicit mutual exclusion, they access and modify the same memory cell simultaneously.
    </div>

    <div class="analysis-p">
        <strong>2. Non-Atomic Read-Modify-Write Operation:</strong><br>
        At the hardware Instruction Set Architecture (ISA) level, the increment expression <code>shared_counter += 1</code> is not executed as an indivisible atomic primitive. Instead, the compiler decomposes it into three distinct sequential machine instructions:
        <ol>
            <li><strong>Read (Load):</strong> Load <code>shared_counter</code> from L1/L2 cache or RAM into a thread-private register (<code>movq shared_counter(%rip), %rax</code>).</li>
            <li><strong>Modify (Increment):</strong> Increment the value within the local CPU register (<code>addq $1, %rax</code>).</li>
            <li><strong>Write (Store):</strong> Write the updated value back from the register to memory (<code>movq %rax, shared_counter(%rip)</code>).</li>
        </ol>
    </div>

    <div class="analysis-p">
        <strong>3. Instruction Interleaving and Lost Updates:</strong><br>
        Because thread dispatch and execution speeds across multi-core CPUs are asynchronously scheduled, the 3-step sequences of different threads interleave arbitrarily. If Thread 0 and Thread 1 both read <code>shared_counter = 100</code> into their respective registers before either writes back, both will calculate <code>101</code> and store <code>101</code> into memory. Although two discrete increments occurred, the counter advances by only 1, resulting in a <strong>lost update</strong>.
    </div>

    <div class="analysis-p">
        <strong>4. Empirical Evaluation:</strong><br>
        In our benchmark execution with 4 threads performing 50,000 increments each:
        <ul>
            <li><strong>Expected Deterministic Value:</strong> 4 &times; 50,000 = <strong>200,000</strong>.</li>
            <li><strong>Actual Measured Value:</strong> <strong>53,739</strong> (~73.13% data loss).</li>
        </ul>
        This severe discrepancy empirically demonstrates that unsynchronized concurrent writes in shared-memory environments lead to corrupted state and non-deterministic execution.
    </div>

    <div class="analysis-p">
        <strong>5. Complexity Breakdown:</strong><br>
        <ul>
            <li><strong>Total Work:</strong> O(P &times; ITERATIONS) arithmetic operations.</li>
            <li><strong>Parallel Time Complexity:</strong> O((P &times; ITERATIONS) / P) = O(ITERATIONS) wall-clock time.</li>
            <li><strong>Auxiliary Space Complexity:</strong> O(1) auxiliary memory.</li>
        </ul>
    </div>
</div>

<!-- ======================================================================= -->
<!-- PROBLEM 2 -->
<!-- ======================================================================= -->

<div class="page-break"></div>

<div class="header-rule">
    <div>Name: <u>Shaik Jnaved</u></div>
    <div>Enrollment No.: <u>P26DS001</u></div>
</div>

<div class="center-title">
    <div class="dept">Computer Science and Engineering Department, SVNIT, Surat</div>
    <div class="deg">M.Tech. I -- Semester I</div>
    <div class="sub">High Performance Computing</div>
    <div class="lab">Lab Assignment 1 -- Problem 2</div>
</div>

<div class="section-h">Problem Statement</div>

<div class="prob-title">Title: Race Condition Resolution via OpenMP Critical Section Synchronization</div>
<div class="prob-desc">
Implement an OpenMP critical section based solution to resolve the data race condition observed in Problem 1. Using the <code>#pragma omp critical</code> directive, enforce mutual exclusion over the shared resource updates across 4 concurrent threads executing 50,000 iterations each. Verify that the final actual value matches the expected value of 200,000 exactly, and analyze the architectural trade-offs, serialization penalties, and alternative OpenMP synchronization constructs.
</div>

<div class="section-h">Code:</div>
<div class="code-container">
{format_code_lines(q2_code)}
</div>

<div class="page-break"></div>

<div class="header-rule">
    <div>Name: <u>Shaik Jnaved</u></div>
    <div>Enrollment No.: <u>P26DS001</u></div>
</div>

<div class="section-h">Output:</div>
<div class="figure-box">
    <img src="data:image/png;base64,{q2_img_b64}" alt="Q2 Output Screenshot" />
    <div class="figure-caption">Figure 2: Execution Output for Problem 2: Resolved Race Condition with OpenMP Critical Section</div>
</div>

<div class="section-h">Analysis:</div>
<div class="analysis-body">
    <div class="analysis-p">
        <strong>1. Mutual Exclusion Semantics (<code>#pragma omp critical</code>):</strong><br>
        The <code>#pragma omp critical</code> directive creates an exclusive execution boundary ensuring that only one thread can execute the enclosed code block at any given time. When a thread reaches the critical section:
        <ol>
            <li>It acquires an internal mutual exclusion lock managed by the OpenMP runtime.</li>
            <li>All other competing threads are placed in a waiting (spin-lock or suspended) state until the holding thread exits the block and releases the lock.</li>
            <li>Hardware memory fences ensure that the updated value of <code>shared_counter</code> is committed to cache/memory before the next thread acquires the lock.</li>
        </ol>
    </div>

    <div class="analysis-p">
        <strong>2. Numerical Verification:</strong><br>
        The terminal output confirms complete correctness:
        <div class="math-eq">Actual Final Value = <strong>200,000</strong> &equiv; Expected Value = <strong>200,000</strong></div>
        Every single increment operation was serialized, resulting in 0 lost updates and 100% numerical consistency.
    </div>

    <div class="analysis-p">
        <strong>3. Serialization Penalty & Amdahl's Law Bottleneck:</strong><br>
        While <code>#pragma omp critical</code> guarantees correctness, embedding a critical section inside a high-frequency loop introduces significant overhead:
        <ul>
            <li><strong>Lock Overhead:</strong> Constant acquisition and release of locks incurs CPU bus arbitration cycles.</li>
            <li><strong>Serialization:</strong> Because the parallel region spends nearly 100% of its time inside the critical block, only 1 core executes at a time. Amdahl's Law dictates that speedup S &le; 1 (often slower than sequential due to locking overhead).</li>
            <li><strong>Cache Invalidation:</strong> Rapid alternating writes across CPU cores trigger cache coherency invalidation traffic (MESI protocol).</li>
        </ul>
    </div>

    <div class="analysis-p">
        <strong>4. Comparison of OpenMP Synchronization Primitives:</strong>
        <table class="academic-table">
            <thead>
                <tr>
                    <th>Construct</th>
                    <th>Mechanism</th>
                    <th>Lock Overhead</th>
                    <th>Parallel Speedup</th>
                    <th>Optimal Use Case</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>#pragma omp critical</code></td>
                    <td>General Mutex Lock</td>
                    <td>High (lock/unlock per op)</td>
                    <td>Low (S &approx; 1)</td>
                    <td>Multi-statement complex blocks</td>
                </tr>
                <tr>
                    <td><code>#pragma omp atomic</code></td>
                    <td>Hardware Bus Lock</td>
                    <td>Moderate (single opcode)</td>
                    <td>Moderate</td>
                    <td>Single memory location updates</td>
                </tr>
                <tr>
                    <td><code>#pragma omp reduction(+:var)</code></td>
                    <td>Private accum. + tree merge</td>
                    <td>Zero inside loop</td>
                    <td>High (S &approx; P)</td>
                    <td>Mathematical summations / reductions</td>
                </tr>
                <tr>
                    <td>Thread-Private + Master Merge</td>
                    <td>Local variable reduction</td>
                    <td>Minimal</td>
                    <td>High (S &approx; P)</td>
                    <td>Custom aggregation routines</td>
                </tr>
            </tbody>
        </table>
    </div>
</div>

<!-- ======================================================================= -->
<!-- PROBLEM 3 -->
<!-- ======================================================================= -->

<div class="page-break"></div>

<div class="header-rule">
    <div>Name: <u>Shaik Jnaved</u></div>
    <div>Enrollment No.: <u>P26DS001</u></div>
</div>

<div class="center-title">
    <div class="dept">Computer Science and Engineering Department, SVNIT, Surat</div>
    <div class="deg">M.Tech. I -- Semester I</div>
    <div class="sub">High Performance Computing</div>
    <div class="lab">Lab Assignment 1 -- Problem 3</div>
</div>

<div class="section-h">Problem Statement</div>

<div class="prob-title">Title: Parallel Matrix Multiplication and Data Decomposition Granularity Analysis</div>
<div class="prob-desc">
Design and develop a parallel OpenMP solution for dense square matrix multiplication (C = A &times; B) of dimension N = 1500 across 8 concurrent execution threads. Implement, benchmark, and contrast the performance of two distinct data decomposition strategies:
<ol>
    <li><strong>Coarse-Grained Decomposition:</strong> 1D row-wise decomposition partitioning outer loop iterations across threads using <code>#pragma omp parallel for private(j, k)</code>.</li>
    <li><strong>Fine-Grained Decomposition:</strong> 2D element-wise decomposition collapsing the two outer loop indices into a unified 2D iteration space using <code>#pragma omp parallel for collapse(2)</code>.</li>
</ol>
Evaluate the empirical execution times, analyze memory access patterns, cache spatial locality, and thread scheduling overheads.
</div>

<div class="section-h">Code:</div>
<div class="code-container">
{format_code_lines(q3_code)}
</div>

<div class="page-break"></div>

<div class="header-rule">
    <div>Name: <u>Shaik Jnaved</u></div>
    <div>Enrollment No.: <u>P26DS001</u></div>
</div>

<div class="section-h">Output:</div>
<div class="figure-box">
    <img src="data:image/png;base64,{q3_img_b64}" alt="Q3 Timing Benchmark" />
    <div class="figure-caption">Figure 3: Execution Timing Benchmark for Problem 3 (N = 1500, Threads = 8)</div>
</div>

<div class="figure-box">
    <img src="data:image/png;base64,{q3_diag_b64}" alt="Granularity Decomposition Comparison" />
    <div class="figure-caption">Figure 4: Data Decomposition Architecture: Coarse-Grained (1D) vs. Fine-Grained (2D Collapsed)</div>
</div>

<div class="section-h">Analysis:</div>
<div class="analysis-body">
    <div class="analysis-p">
        <strong>1. Mathematical Model of Matrix Multiplication:</strong><br>
        For dense square matrices A, B &isin; &reals;<sup>N&times;N</sup>, each element of the product matrix C = A &times; B is evaluated as:
        <div class="math-eq">C<sub>ij</sub> = &sum;<sub>k=0</sub><sup>N-1</sup> A<sub>ik</sub> &middot; B<sub>kj</sub> &emsp; (&forall; 0 &le; i, j &lt; N)</div>
        Computing each cell requires N multiplications and N-1 additions (O(N) operations). For N = 1500, the full matrix requires N<sup>3</sup> = 3.375 &times; 10<sup>9</sup> multiply-accumulate operations (O(N<sup>3</sup>) total complexity).
    </div>

    <div class="analysis-p">
        <strong>2. Coarse-Grained Decomposition (1D Row-Wise Partitioning):</strong><br>
        Using <code>#pragma omp parallel for private(j, k) num_threads(8)</code>, only the outermost loop (i) is partitioned across the 8 threads. The N = 1500 rows of matrix C are assigned in large contiguous blocks of &lceil;1500 / 8&rceil; &approx; 187 rows per thread.
        <ul>
            <li><strong>Task Unit:</strong> Entire row computations (large grain size).</li>
            <li><strong>Scheduling Overhead:</strong> Minimal, as only 1500 loop iterations are managed by the OpenMP runtime.</li>
        </ul>
    </div>

    <div class="analysis-p">
        <strong>3. Fine-Grained Decomposition (2D Element-Wise Partitioning):</strong><br>
        Using <code>#pragma omp parallel for collapse(2) num_threads(8)</code>, the compiler flattens the two outer loops (i and j) into a single unified 1D loop of size N<sup>2</sup> = 2,250,000 iterations.
        <ul>
            <li><strong>Task Unit:</strong> Individual element computation C<sub>ij</sub> (small grain size).</li>
            <li><strong>Scheduling Overhead:</strong> Substantial, requiring 2.25 million task dispatches and 2D index division/modulo mappings at runtime.</li>
        </ul>
    </div>

    <div class="analysis-p">
        <strong>4. Cache Spatial Locality & Memory Hierarchy:</strong><br>
        In C, 2D arrays are stored in contiguous <strong>row-major order</strong>:
        <ul>
            <li><strong>Matrix A Access (A[i][k]):</strong> As k increments, memory is accessed contiguously. Loading one cache line (64 bytes = 16 integers) yields 15 subsequent L1 cache hits (<strong>high spatial locality</strong>).</li>
            <li><strong>Matrix B Access (B[k][j]):</strong> As k increments, index jumps by N &times; 4 bytes = 6000 bytes, jumping cache lines on every operation (<strong>poor spatial locality</strong>).</li>
            <li><strong>Matrix C Write (C[i][j]):</strong> Coarse-grained keeps each thread operating on distinct contiguous row blocks, completely preventing <strong>false sharing</strong> across CPU cores.</li>
        </ul>
    </div>

    <div class="analysis-p">
        <strong>5. Empirical Results & Performance Verdict:</strong><br>
        <ul>
            <li><strong>Coarse-Grained Execution Time:</strong> <strong>0.345000 seconds</strong></li>
            <li><strong>Fine-Grained Execution Time:</strong> <strong>0.629000 seconds</strong></li>
            <li><strong>Speedup Ratio:</strong> 0.629000 / 0.345000 &approx; <strong>1.82&times; speedup</strong> in favor of coarse-grained decomposition.</li>
        </ul>
    </div>

    <table class="academic-table">
        <thead>
            <tr>
                <th>Strategy</th>
                <th>OpenMP Directive</th>
                <th>Task Unit</th>
                <th>Total Tasks</th>
                <th>Scheduling Overhead</th>
                <th>Execution Time (N=1500)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Coarse-Grained (1D)</td>
                <td><code>parallel for private(j,k)</code></td>
                <td>Entire Row</td>
                <td>1,500</td>
                <td>Low</td>
                <td><strong>0.345000 s</strong></td>
            </tr>
            <tr>
                <td>Fine-Grained (2D)</td>
                <td><code>parallel for collapse(2)</code></td>
                <td>Single Cell C<sub>ij</sub></td>
                <td>2,250,000</td>
                <td>High</td>
                <td><strong>0.629000 s</strong></td>
            </tr>
            <tr>
                <td>Sequential (1 Thread)</td>
                <td>Standard nested loops</td>
                <td>Full Matrix</td>
                <td>1</td>
                <td>None</td>
                <td>&approx; 2.450 s</td>
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
    
    # Generate in lab_1 and root
    pdf_lab1 = base_dir / "lab_1" / "lab_assignment.pdf"
    pdf_root = base_dir / "lab_assignment.pdf"
    
    convert_html_to_pdf(html, pdf_lab1)
    shutil.copy(pdf_lab1, pdf_root)
    print(f"[SUCCESS] Lab Assignment PDF generated successfully at:\n  - {pdf_lab1}\n  - {pdf_root}")

if __name__ == "__main__":
    main()
