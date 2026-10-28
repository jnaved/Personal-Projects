import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

# Styling configuration for academic presentation
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2.0

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. TASK ALLOCATION PROBLEM (TAP): RUNTIME & SPEEDUP COMPARISON
# ==============================================================================
tap_file = BASE_DIR / "tap_results.csv"
if tap_file.exists():
    tap_data = {}
    with open(tap_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alg = row['Algorithm']
            ver = row['Version']
            if alg not in tap_data:
                tap_data[alg] = {}
            tap_data[alg][ver] = {
                'Time': float(row['Time_Sec']),
                'Cost': float(row['Best_Cost']),
                'Speedup': float(row['Speedup'])
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    algorithms = ['GA', 'PSO']
    x = np.arange(len(algorithms))
    width = 0.25

    # Left: Execution Time (seconds)
    t_seq = [tap_data[alg]['Sequential']['Time'] for alg in algorithms]
    t_omp = [tap_data[alg]['OpenMP']['Time'] for alg in algorithms]
    t_cuda = [tap_data[alg]['CUDA']['Time'] for alg in algorithms]

    rects1 = ax1.bar(x - width, t_seq, width, label='Sequential C', color='#d62728', alpha=0.88, edgecolor='black')
    rects2 = ax1.bar(x, t_omp, width, label='OpenMP Parallel (16 th)', color='#1f77b4', alpha=0.88, edgecolor='black')
    rects3 = ax1.bar(x + width, t_cuda, width, label='CUDA C (RTX 4050 GPU)', color='#2ca02c', alpha=0.88, edgecolor='black')

    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('(a) Task Allocation Problem: Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Genetic Algorithm (GA)', 'Particle Swarm (PSO)'])
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.3f}s',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, weight='bold')

    # Right: Speedup Factor
    sp_seq = [1.0, 1.0]
    sp_omp = [tap_data[alg]['OpenMP']['Speedup'] for alg in algorithms]
    sp_cuda = [tap_data[alg]['CUDA']['Speedup'] for alg in algorithms]

    rects4 = ax2.bar(x - width, sp_seq, width, label='Sequential Baseline (1.0x)', color='#d62728', alpha=0.88, edgecolor='black')
    rects5 = ax2.bar(x, sp_omp, width, label='OpenMP Parallel', color='#1f77b4', alpha=0.88, edgecolor='black')
    rects6 = ax2.bar(x + width, sp_cuda, width, label='CUDA C GPU', color='#2ca02c', alpha=0.88, edgecolor='black')

    ax2.set_ylabel('Speedup Factor ($T_{seq} / T_{par}$)')
    ax2.set_title('(b) Task Allocation Problem: Measured Speedup')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Genetic Algorithm (GA)', 'Particle Swarm (PSO)'])
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    for rects in [rects4, rects5, rects6]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.2f}x',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "tap_performance_comparison.png", bbox_inches='tight')
    plt.close()
    print("Saved tap_performance_comparison.png")

# ==============================================================================
# 2. FIVE BENCHMARK FUNCTIONS: RUNTIME & SPEEDUP COMPARISON
# ==============================================================================
bench_file = BASE_DIR / "benchmark_results.csv"
if bench_file.exists():
    funcs = []
    pso_seq = []
    pso_omp = []
    pso_cuda = []
    pso_sp_omp = []
    pso_sp_cuda = []

    ga_seq = []
    ga_omp = []
    ga_cuda = []
    ga_sp_omp = []
    ga_sp_cuda = []

    with open(bench_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row['Function']
            alg = row['Algorithm']
            if alg == 'PSO':
                funcs.append(fn)
                pso_seq.append(float(row['Sequential_Time']))
                pso_omp.append(float(row['OpenMP_Time']))
                pso_cuda.append(float(row['CUDA_Time']))
                pso_sp_omp.append(float(row['Speedup_OpenMP']))
                pso_sp_cuda.append(float(row['Speedup_CUDA']))
            elif alg == 'GA':
                ga_seq.append(float(row['Sequential_Time']))
                ga_omp.append(float(row['OpenMP_Time']))
                ga_cuda.append(float(row['CUDA_Time']))
                ga_sp_omp.append(float(row['Speedup_OpenMP']))
                ga_sp_cuda.append(float(row['Speedup_CUDA']))

    x = np.arange(len(funcs))
    width = 0.26

    # Figure 2: Benchmark Execution Time Comparison (GA & PSO)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: PSO Runtimes
    ax1.bar(x - width, pso_seq, width, label='Sequential PSO', color='#d62728', alpha=0.88, edgecolor='black')
    ax1.bar(x, pso_omp, width, label='OpenMP PSO (16 th)', color='#1f77b4', alpha=0.88, edgecolor='black')
    ax1.bar(x + width, pso_cuda, width, label='CUDA C PSO (GPU)', color='#2ca02c', alpha=0.88, edgecolor='black')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('(a) PSO: Execution Time on 5 Benchmark Functions')
    ax1.set_xticks(x)
    ax1.set_xticklabels(funcs)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

    # Right: GA Runtimes
    ax2.bar(x - width, ga_seq, width, label='Sequential GA', color='#d62728', alpha=0.88, edgecolor='black')
    ax2.bar(x, ga_omp, width, label='OpenMP GA (16 th)', color='#1f77b4', alpha=0.88, edgecolor='black')
    ax2.bar(x + width, ga_cuda, width, label='CUDA C GA (GPU)', color='#2ca02c', alpha=0.88, edgecolor='black')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('(b) GA: Execution Time on 5 Benchmark Functions')
    ax2.set_xticks(x)
    ax2.set_xticklabels(funcs)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "benchmark_runtime_comparison.png", bbox_inches='tight')
    plt.close()
    print("Saved benchmark_runtime_comparison.png")

    # Figure 3: Benchmark Speedup Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    width2 = 0.35
    # Left: PSO Speedups
    ax1.bar(x - width2/2, pso_sp_omp, width2, label='OpenMP Speedup', color='#1f77b4', alpha=0.88, edgecolor='black')
    ax1.bar(x + width2/2, pso_sp_cuda, width2, label='CUDA C Speedup', color='#2ca02c', alpha=0.88, edgecolor='black')
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='Sequential Baseline (1.0x)')
    ax1.set_ylabel('Speedup Factor ($T_{seq} / T_{par}$)')
    ax1.set_title('(a) PSO Parallel Speedup Across Benchmarks')
    ax1.set_xticks(x)
    ax1.set_xticklabels(funcs)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

    for i in range(len(funcs)):
        ax1.annotate(f"{pso_sp_omp[i]:.1f}x", (x[i] - width2/2, pso_sp_omp[i]),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, weight='bold')
        ax1.annotate(f"{pso_sp_cuda[i]:.1f}x", (x[i] + width2/2, pso_sp_cuda[i]),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, weight='bold')

    # Right: GA Speedups
    ax2.bar(x - width2/2, ga_sp_omp, width2, label='OpenMP Speedup', color='#1f77b4', alpha=0.88, edgecolor='black')
    ax2.bar(x + width2/2, ga_sp_cuda, width2, label='CUDA C Speedup', color='#2ca02c', alpha=0.88, edgecolor='black')
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='Sequential Baseline (1.0x)')
    ax2.set_ylabel('Speedup Factor ($T_{seq} / T_{par}$)')
    ax2.set_title('(b) GA Parallel Speedup Across Benchmarks')
    ax2.set_xticks(x)
    ax2.set_xticklabels(funcs)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    for i in range(len(funcs)):
        ax2.annotate(f"{ga_sp_omp[i]:.1f}x", (x[i] - width2/2, ga_sp_omp[i]),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, weight='bold')
        ax2.annotate(f"{ga_sp_cuda[i]:.1f}x", (x[i] + width2/2, ga_sp_cuda[i]),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, weight='bold')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "benchmark_speedup_comparison.png", bbox_inches='tight')
    plt.close()
    print("Saved benchmark_speedup_comparison.png")
