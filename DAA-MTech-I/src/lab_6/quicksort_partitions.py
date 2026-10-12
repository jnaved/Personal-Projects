"""
Assignment 3:
Observe that the algorithm PARTITION() for Quicksort given in the text is different from the
one discussed in the class. Implement two different Quicksort routines that use the two different
PARTITION routines. Test your implementation on an input vector of size at least 2048
integers and time the programs. Analyze the result and document it.
"""

import time
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Increase recursion depth for deep Quicksort recursion if needed
sys.setrecursionlimit(200_000)

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent.parent
IMAGES_DIRS = [
    WORKSPACE_DIR / "images",
    WORKSPACE_DIR / "overleaf" / "lab_6" / "images",
]
for d in IMAGES_DIRS:
    d.mkdir(parents=True, exist_ok=True)


class SortStats:
    """Tracks comparisons and swaps during sorting."""
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0


# ==============================================================================
# 1. LOMUTO PARTITION (CLRS Textbook Scheme)
# ==============================================================================
def lomuto_partition(arr: list[int], low: int, high: int, stats: SortStats | None = None) -> int:
    """
    Lomuto Partition (Textbook Scheme):
    - Pivot is chosen as the last element arr[high].
    - Scans with single index j from low to high - 1.
    - Maintains arr[low ... i] <= pivot and arr[i+1 ... j-1] > pivot.
    - Places the pivot at its exact final sorted index (i + 1).
    """
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if stats:
            stats.comparisons += 1
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            if stats:
                stats.swaps += 1

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    if stats:
        stats.swaps += 1

    return i + 1


def quicksort_lomuto(arr: list[int], low: int, high: int, stats: SortStats | None = None):
    """Quicksort routine using Lomuto's Partition."""
    if low < high:
        pi = lomuto_partition(arr, low, high, stats)
        quicksort_lomuto(arr, low, pi - 1, stats)
        quicksort_lomuto(arr, pi + 1, high, stats)


# ==============================================================================
# 2. HOARE PARTITION (Classic Scheme Discussed in Class)
# ==============================================================================
def hoare_partition(arr: list[int], low: int, high: int, stats: SortStats | None = None) -> int:
    """
    Hoare Partition (Classic Scheme):
    - Pivot is chosen as arr[low] (or middle element).
    - Two pointers i and j move toward each other from opposite ends.
    - Stops when arr[i] >= pivot and arr[j] <= pivot, then swaps them.
    - Returns split index j such that arr[low ... j] <= arr[j+1 ... high].
    - Pivot is not necessarily in its final position, but splits subproblems.
    - Typically performs ~3x fewer swaps on average than Lomuto.
    """
    # Use middle element as pivot to prevent worst-case on already sorted inputs
    mid = low + (high - low) // 2
    pivot = arr[mid]

    i = low - 1
    j = high + 1

    while True:
        while True:
            i += 1
            if stats:
                stats.comparisons += 1
            if arr[i] >= pivot:
                break

        while True:
            j -= 1
            if stats:
                stats.comparisons += 1
            if arr[j] <= pivot:
                break

        if i >= j:
            return j

        arr[i], arr[j] = arr[j], arr[i]
        if stats:
            stats.swaps += 1


def quicksort_hoare(arr: list[int], low: int, high: int, stats: SortStats | None = None):
    """Quicksort routine using Hoare's Partition."""
    if low < high:
        pi = hoare_partition(arr, low, high, stats)
        quicksort_hoare(arr, low, pi, stats)
        quicksort_hoare(arr, pi + 1, high, stats)


# ==============================================================================
# BENCHMARK & ANALYSIS DRIVER
# ==============================================================================
def benchmark_quicksort_partitions(show_plot: bool = True):
    """
    Tests both partition schemes on input vectors of size at least 2048 integers.
    Compares execution time, comparisons, and swaps.
    """
    # Vector sizes starting at 2048 as specified in Assignment 3
    sizes = [2048, 4096, 8192, 16384, 32768, 65536]

    lomuto_times = {}
    hoare_times = {}

    lomuto_swaps = {}
    hoare_swaps = {}

    lomuto_comps = {}
    hoare_comps = {}

    rng = random.Random(42)

    print("=" * 95)
    print(f"{'Size (N)':<10} | {'Lomuto (ms)':<14} | {'Hoare (ms)':<14} | {'Lomuto Swaps':<14} | {'Hoare Swaps':<14} | {'Swap Ratio':<10}")
    print("=" * 95)

    for n in sizes:
        base_arr = [rng.randint(1, 1_000_000) for _ in range(n)]

        # Benchmark Lomuto
        arr_lomuto = base_arr.copy()
        stats_lomuto = SortStats()
        t0 = time.perf_counter()
        quicksort_lomuto(arr_lomuto, 0, len(arr_lomuto) - 1, stats_lomuto)
        t_lomuto = (time.perf_counter() - t0) * 1000
        assert arr_lomuto == sorted(base_arr), f"Lomuto failed to sort size {n}"

        # Benchmark Hoare
        arr_hoare = base_arr.copy()
        stats_hoare = SortStats()
        t0 = time.perf_counter()
        quicksort_hoare(arr_hoare, 0, len(arr_hoare) - 1, stats_hoare)
        t_hoare = (time.perf_counter() - t0) * 1000
        assert arr_hoare == sorted(base_arr), f"Hoare failed to sort size {n}"

        lomuto_times[n] = t_lomuto
        hoare_times[n] = t_hoare

        lomuto_swaps[n] = stats_lomuto.swaps
        hoare_swaps[n] = stats_hoare.swaps

        lomuto_comps[n] = stats_lomuto.comparisons
        hoare_comps[n] = stats_hoare.comparisons

        swap_ratio = stats_lomuto.swaps / max(1, stats_hoare.swaps)

        print(f"{n:<10} | {t_lomuto:<14.3f} | {t_hoare:<14.3f} | {stats_lomuto.swaps:<14} | {stats_hoare.swaps:<14} | {swap_ratio:<10.2f}x")

    print("=" * 95)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Subplot 1: Execution Time
    ax1.plot(sizes, [lomuto_times[s] for s in sizes], marker="o", color="#d62728", linewidth=2.2, label="Lomuto Quicksort (Textbook)")
    ax1.plot(sizes, [hoare_times[s] for s in sizes], marker="s", color="#1f77b4", linewidth=2.2, label="Hoare Quicksort (Class/Classic)")
    ax1.set_title("Quicksort Execution Time vs. Vector Size (N ≥ 2048)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Vector Size (N)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=10)

    # Subplot 2: Swaps Incurred
    ax2.plot(sizes, [lomuto_swaps[s] for s in sizes], marker="o", color="#d62728", linewidth=2.2, label="Lomuto Partition Swaps")
    ax2.plot(sizes, [hoare_swaps[s] for s in sizes], marker="s", color="#2ca02c", linewidth=2.2, label="Hoare Partition Swaps")
    ax2.set_title("Number of Swaps Incurred (Lomuto vs. Hoare)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Vector Size (N)", fontsize=11)
    ax2.set_ylabel("Total Swaps", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(fontsize=10)

    plt.suptitle("Quicksort Partition Routine Comparison: Lomuto (Textbook) vs. Hoare (Class)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    for d in IMAGES_DIRS:
        save_path = d / "quicksort_partition_analysis.png"
        plt.savefig(save_path, dpi=300)
        print(f"[Plot Saved] {save_path}")

    if show_plot:
        plt.show()

    return lomuto_times, hoare_times


if __name__ == "__main__":
    benchmark_quicksort_partitions(show_plot=False)
