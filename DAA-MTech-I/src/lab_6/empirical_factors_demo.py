"""
Assignment 6:
Do some research and find out what factors the actual time a program takes depends on.
Use these factors as an argument to justify why it is not feasible to do empirical
analysis when the goal is to compare two algorithms.
"""

import time
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent.parent
IMAGES_DIRS = [
    WORKSPACE_DIR / "images",
    WORKSPACE_DIR / "overleaf" / "lab_6" / "images",
]
for d in IMAGES_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def deterministic_task(n: int = 50_000) -> int:
    """A strictly deterministic calculation with zero random branch divergence."""
    s = 0
    for i in range(n):
        s = (s + i * 3) ^ (i & 0xFF)
    return s


def demonstrate_timing_instability(runs: int = 1000, show_plot: bool = True):
    """
    Empirically demonstrates wall-clock runtime variance across 1,000 identical runs
    of the exact same deterministic program on identical inputs.
    """
    times_us = []

    for _ in range(runs):
        t0 = time.perf_counter_ns()
        deterministic_task()
        times_us.append((time.perf_counter_ns() - t0) / 1000.0)

    min_t = min(times_us)
    max_t = max(times_us)
    mean_t = statistics.mean(times_us)
    std_t = statistics.stdev(times_us)
    median_t = statistics.median(times_us)

    print("=" * 70)
    print("EMPIRICAL TIMING VARIABILITY EXPERIMENT (1,000 IDENTICAL RUNS)")
    print("=" * 70)
    print(f"Minimum Execution Time:  {min_t:.2f} µs")
    print(f"Maximum Execution Time:  {max_t:.2f} µs (Spike factor: {max_t / min_t:.2f}x)")
    print(f"Mean Execution Time:     {mean_t:.2f} µs")
    print(f"Median Execution Time:   {median_t:.2f} µs")
    print(f"Standard Deviation:      {std_t:.2f} µs")
    print("=" * 70)

    # Plot histogram and timeline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Run timeline showing random spikes
    ax1.plot(range(1, runs + 1), times_us, color="#1f77b4", linewidth=0.8, alpha=0.8)
    ax1.axhline(mean_t, color="#d62728", linestyle="--", linewidth=1.5, label=f"Mean: {mean_t:.1f} µs")
    ax1.set_title("Runtime Fluctuation Across Identical Executions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Run Number (Consecutive)", fontsize=11)
    ax1.set_ylabel("Execution Time (µs)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Subplot 2: Distribution histogram
    ax2.hist(times_us, bins=40, color="#2ca02c", edgecolor="black", alpha=0.8)
    ax2.axvline(mean_t, color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean: {mean_t:.1f} µs")
    ax2.axvline(median_t, color="#9467bd", linestyle=":", linewidth=1.8, label=f"Median: {median_t:.1f} µs")
    ax2.set_title("Execution Time Probability Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Execution Time (µs)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.suptitle("Why Empirical Analysis is Infeasible for Comparing Algorithms: Environmental Noise", fontsize=13, fontweight="bold")
    plt.tight_layout()

    for d in IMAGES_DIRS:
        save_path = d / "empirical_timing_variance.png"
        plt.savefig(save_path, dpi=300)
        print(f"[Plot Saved] {save_path}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    demonstrate_timing_instability(show_plot=False)
