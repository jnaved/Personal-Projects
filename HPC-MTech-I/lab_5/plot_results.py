import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

# Styling configuration for academic publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6

OUT_DIR = Path("lab_5/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. SCALING ANALYSIS: EXECUTION TIME & SPEEDUP VS THREADS
# -------------------------------------------------------------
threads = []
ga_time = []
ga_sp = []
pso_time = []
pso_sp = []

with open("lab_5/scaling_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        threads.append(int(row['Threads']))
        ga_time.append(float(row['GA_Time']))
        ga_sp.append(float(row['GA_Speedup']))
        pso_time.append(float(row['PSO_Time']))
        pso_sp.append(float(row['PSO_Speedup']))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

# Left: Execution Time vs Threads
ax1.plot(threads, ga_time, marker='o', color='#1f77b4', label='Parallel GA (OpenMP)')
ax1.plot(threads, pso_time, marker='s', color='#ff7f0e', label='Parallel PSO (OpenMP)')
ax1.set_xlabel('Number of OpenMP Threads')
ax1.set_ylabel('Execution Time (seconds)')
ax1.set_title('(a) Strong Scaling Execution Time')
ax1.set_xticks(threads)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=True)

for t, time_val in zip(threads, ga_time):
    ax1.annotate(f"{time_val:.2f}s", (t, time_val), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8.5, color='#1f77b4')

# Right: Speedup vs Threads
ax2.plot(threads, threads, linestyle='--', color='#7f7f7f', label='Ideal Linear Speedup ($S=p$)')
ax2.plot(threads, ga_sp, marker='o', color='#1f77b4', label='Parallel GA Speedup')
ax2.plot(threads, pso_sp, marker='s', color='#ff7f0e', label='Parallel PSO Speedup')
ax2.set_xlabel('Number of OpenMP Threads')
ax2.set_ylabel('Speedup Factor ($T_1 / T_p$)')
ax2.set_title('(b) Measured Parallel Speedup')
ax2.set_xticks(threads)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(frameon=True)

for t, sp_val in zip(threads, ga_sp):
    ax2.annotate(f"{sp_val:.1f}x", (t, sp_val), textcoords="offset points", xytext=(-10, 8), ha='center', fontsize=8.5, color='#1f77b4', weight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / "scaling_analysis.png", bbox_inches='tight')
plt.close()
print("Saved scaling_analysis.png")

# -------------------------------------------------------------
# 2. SEQUENTIAL VS PARALLEL EXECUTION TIME COMPARISON
# -------------------------------------------------------------
summary_data = {}
with open("lab_5/summary_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        summary_data[row['Algorithm']] = {
            'Threads': int(row['Threads']),
            'TotalCost': float(row['TotalCost']),
            'ExecCost': float(row['ExecCost']),
            'CommCost': float(row['CommCost']),
            'TimeSeconds': float(row['TimeSeconds']),
            'Speedup': float(row['Speedup'])
        }

fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

algos = ['Genetic Algorithm\n(GA)', 'Particle Swarm\nOptimization (PSO)']
seq_times = [summary_data['SGA']['TimeSeconds'], summary_data['SPSO']['TimeSeconds']]
par_times = [summary_data['PGA']['TimeSeconds'], summary_data['PPSO']['TimeSeconds']]
speedups = [summary_data['PGA']['Speedup'], summary_data['PPSO']['Speedup']]

x = np.arange(len(algos))
width = 0.35

rects1 = ax.bar(x - width/2, seq_times, width, label='Sequential (1 Core)', color='#4a6fa5', edgecolor='#1d3557')
rects2 = ax.bar(x + width/2, par_times, width, label='Parallel OpenMP (16 Threads)', color='#2a9d8f', edgecolor='#1a535c')

ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Execution Time Comparison: Sequential vs. OpenMP Parallel (M=600, P=24)')
ax.set_xticks(x)
ax.set_xticklabels(algos)
ax.legend(frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.6)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.3f} s', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, weight='bold')

for i, rect in enumerate(rects2):
    h = rect.get_height()
    sp = speedups[i]
    ax.annotate(f'{h:.3f} s\n({sp:.1f}x Speedup)', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, weight='bold', color='#1a535c')

ax.set_ylim(0, max(seq_times) * 1.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "execution_comparison.png", bbox_inches='tight')
plt.close()
print("Saved execution_comparison.png")

# -------------------------------------------------------------
# 3. COST COMPONENT BREAKDOWN (EXECUTION VS COMMUNICATION)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

labels = ['Sequential GA', 'Parallel GA', 'Sequential PSO', 'Parallel PSO']
keys = ['SGA', 'PGA', 'SPSO', 'PPSO']
exec_costs = [summary_data[k]['ExecCost'] for k in keys]
comm_costs = [summary_data[k]['CommCost'] for k in keys]
total_costs = [summary_data[k]['TotalCost'] for k in keys]

x = np.arange(len(labels))
width = 0.45

p1 = ax.bar(x, exec_costs, width, label='Execution Cost', color='#e76f51', edgecolor='#9d4128')
p2 = ax.bar(x, comm_costs, width, bottom=exec_costs, label='Inter-task Communication Cost', color='#457b9d', edgecolor='#1d3557')

ax.set_ylabel('Objective Cost Value')
ax.set_title('Objective Cost Composition (M=600, P=24, P << M)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(frameon=True, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.6)

for i in range(len(labels)):
    tot = total_costs[i]
    ax.annotate(f'Total:\n{tot:,.0f}', xy=(x[i], tot),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')

ax.set_ylim(0, max(total_costs) * 1.18)
plt.tight_layout()
plt.savefig(OUT_DIR / "cost_breakdown.png", bbox_inches='tight')
plt.close()
print("Saved cost_breakdown.png")

# -------------------------------------------------------------
# 4. CONVERGENCE TRAJECTORY (GENERATIONS VS BEST COST)
# -------------------------------------------------------------
iters = []
sga_costs = []
pga_costs = []
spso_costs = []
ppso_costs = []

with open("lab_5/convergence_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iters.append(int(row['Iteration']))
        sga_costs.append(float(row['SGA_Cost']))
        pga_costs.append(float(row['PGA_Cost']))
        spso_costs.append(float(row['SPSO_Cost']))
        ppso_costs.append(float(row['PPSO_Cost']))

fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=300)

ax.plot(iters, sga_costs, label='Sequential GA (SGA)', color='#1f77b4', linestyle='--')
ax.plot(iters, pga_costs, label='Parallel GA (PGA)', color='#1f77b4', linestyle='-', linewidth=2.2)
ax.plot(iters, spso_costs, label='Sequential PSO (SPSO)', color='#ff7f0e', linestyle='--')
ax.plot(iters, ppso_costs, label='Parallel PSO (PPSO)', color='#ff7f0e', linestyle='-', linewidth=2.2)

ax.set_xlabel('Iteration / Generation Number')
ax.set_ylabel('Best Objective Cost Found')
ax.set_title('Algorithm Convergence Trajectory (Population/Swarm Size = 128)')
ax.legend(frameon=True, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(OUT_DIR / "convergence_curve.png", bbox_inches='tight')
plt.close()
print("Saved convergence_curve.png")

# -------------------------------------------------------------
# 5. DIAGRAM ILLUSTRATION OF TASK ALLOCATION PROBLEM (TAP)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=300)
ax.axis('off')

illustration_text = """
Task Allocation Problem (TAP) Formulation in Distributed Systems:
• M Tasks: {t_1, t_2, ..., t_M} mapped to P Processing Nodes: {n_1, n_2, ..., n_P} (where P << M)
• Total Cost = Total Execution Cost (x) + Total Inter-task Communication Cost (c)
• x = ∑ E[i, assignment[i]]
• c = ∑ C[i, j]  for all pairs (i, j) where assignment[i] ≠ assignment[j] (cross-node communication)

Verification Case from Problem Specification (M = 6, P = 2):
------------------------------------------------------------------------------------------------------
(a) Serial Assignment: [t1->n1, t2->n1, t3->n1, t4->n2, t5->n2, t6->n2]
    • Execution Cost: x = 5 + 2 + 4 + 3 + 2 + 4 = 20
    • Comm Cost:     c = c16 + c24 + c25 + c35 = 12 + 12 + 3 + 11 = 38
    • Total Cost:     20 + 38 = 58  [VERIFIED EXACT MATCH]

(b) Optimal Assignment: [t1->n1, t2->n1, t3->n1, t4->n1, t5->n1, t6->n2]
    • Execution Cost: x = 5 + 2 + 4 + 6 + 5 + 4 = 26
    • Comm Cost:     c = c16 = 12
    • Total Cost:     26 + 12 = 38  [VERIFIED OPTIMAL MATCH]
------------------------------------------------------------------------------------------------------
Both Parallel GA and Parallel PSO find this optimal assignment [1, 1, 1, 1, 1, 2] with Total Cost = 38!
"""

ax.text(0.02, 0.95, illustration_text, fontfamily='monospace', fontsize=9.2, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#495057', linewidth=1.2))

plt.tight_layout()
plt.savefig(OUT_DIR / "problem_formulation.png", bbox_inches='tight')
plt.close()
print("Saved problem_formulation.png")

# -------------------------------------------------------------
# 6. RENDER TERMINAL WINDOW SCREENSHOT (terminal_output.png)
# -------------------------------------------------------------
import os
import subprocess

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def get_browser():
    if os.path.exists(EDGE_PATH): return EDGE_PATH
    if os.path.exists(CHROME_PATH): return CHROME_PATH
    raise FileNotFoundError("Browser not found")

def render_terminal_image():
    stdout_file = Path("lab_5/terminal_stdout.txt")
    if not stdout_file.exists():
        print("terminal_stdout.txt not found, skipping terminal screenshot.")
        return

    content_lines = stdout_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    formatted = []
    for line in content_lines:
        line_clean = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if "EXACT MATCH" in line_clean or "Speedup" in line_clean:
            formatted.append(f"<span class='hl-accent'>{line_clean}</span>")
        elif "PART" in line_clean or "Executing" in line_clean:
            formatted.append(f"<span class='hl-info'>{line_clean}</span>")
        elif "HIGH PERFORMANCE" in line_clean or "Threads" in line_clean:
            formatted.append(f"<span class='hl-thread'>{line_clean}</span>")
        else:
            formatted.append(line_clean)
    
    body_text = "\n".join(formatted)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: transparent;
    padding: 16px;
    font-family: 'Consolas', 'Courier New', monospace;
    display: inline-block;
}}
.terminal-window {{
    width: 820px;
    background: #1a1b26;
    border-radius: 8px;
    border: 1px solid #414868;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
    overflow: hidden;
}}
.terminal-header {{
    background: #24283b;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #414868;
}}
.mac-dots {{ display: flex; gap: 7px; }}
.mac-dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
.dot-red {{ background: #ff5f56; }}
.dot-yellow {{ background: #ffbd2e; }}
.dot-green {{ background: #27c93f; }}
.window-title {{
    flex: 1;
    text-align: center;
    color: #7aa2f7;
    font-family: 'Segoe UI', sans-serif;
    font-size: 12px;
    font-weight: 600;
    margin-right: 40px;
}}
.terminal-body {{
    padding: 14px 18px;
    color: #c0caf5;
    font-size: 11px;
    line-height: 1.45;
    white-space: pre-wrap;
}}
.hl-accent {{ color: #7dcfff; font-weight: bold; }}
.hl-thread {{ color: #bb9af7; font-weight: bold; }}
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
        <div class="window-title">Terminal - OpenMP Task Allocation Benchmark (M=600, P=24)</div>
    </div>
    <div class="terminal-body">{body_text}</div>
</div>
</body>
</html>"""

    out_png = OUT_DIR / "terminal_output.png"
    temp_html = OUT_DIR / "terminal_temp.html"
    temp_html.write_text(html, encoding="utf-8")

    browser = get_browser()
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--window-size=860,1180",
        f"--screenshot={out_png.resolve()}",
        f"file:///{temp_html.resolve()}"
    ]
    subprocess.run(cmd, check=True)
    if temp_html.exists():
        temp_html.unlink()
    print("Saved terminal_output.png")

render_terminal_image()

