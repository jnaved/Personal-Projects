"""
Assignment 2:
Write programs to count the number of inversions in a given array using the brute force
approach and using the algorithm applying the merge-sort discussed in the class (with time
complexities O(n^2) and O(n lg n) respectively). Compare both the programs with timing
analysis with input size starting with n = 10 upto n = 10^5. Comment on the results obtained.
"""

import time
import random
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


def count_inversions_brute_force(arr: list[int]) -> int:
    """
    Brute-force approach to count inversions.
    Checks all pairs (i, j) with i < j and arr[i] > arr[j].

    Time Complexity: Theta(n^2)
    Space Complexity: O(1)
    """
    n = len(arr)
    inversions = 0
    for i in range(n - 1):
        ai = arr[i]
        for j in range(i + 1, n):
            if ai > arr[j]:
                inversions += 1
    return inversions


def count_inversions_merge_sort(arr: list[int]) -> tuple[list[int], int]:
    """
    Divide-and-conquer approach using enhanced Merge Sort.
    Recursively counts inversions in left and right halves and cross/split inversions.

    Time Complexity: Theta(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr, 0

    mid = len(arr) // 2
    left_sorted, left_inv = count_inversions_merge_sort(arr[:mid])
    right_sorted, right_inv = count_inversions_merge_sort(arr[mid:])

    merged = []
    i, j = 0, 0
    split_inv = 0
    len_left, len_right = len(left_sorted), len(right_sorted)

    while i < len_left and j < len_right:
        if left_sorted[i] <= right_sorted[j]:
            merged.append(left_sorted[i])
            i += 1
        else:
            merged.append(right_sorted[j])
            # Since left_sorted is sorted, all elements from i to len_left-1
            # are greater than right_sorted[j]
            split_inv += (len_left - i)
            j += 1

    merged.extend(left_sorted[i:])
    merged.extend(right_sorted[j:])

    total_inv = left_inv + right_inv + split_inv
    return merged, total_inv


def benchmark_inversion_counting(show_plot: bool = True):
    """
    Benchmarks Brute Force vs Merge Sort counting for input sizes from n=10 up to n=10^5.
    """
    # Test sizes spanning 10 up to 100,000
    sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    brute_force_times = {}
    merge_sort_times = {}
    inversion_counts = {}

    rng = random.Random(42)

    print("=" * 80)
    print(f"{'Size (n)':<10} | {'Inversions':<14} | {'Brute Force (ms)':<18} | {'Merge Sort (ms)':<16} | {'Speedup':<10}")
    print("=" * 80)

    for n in sizes:
        arr = [rng.randint(1, 1_000_000) for _ in range(n)]

        # Benchmark Merge Sort (all sizes up to 10^5)
        iters_ms = 10 if n <= 1000 else (5 if n <= 10000 else 2)
        t_ms_start = time.perf_counter()
        for _ in range(iters_ms):
            _, count_ms = count_inversions_merge_sort(arr)
        t_ms = ((time.perf_counter() - t_ms_start) / iters_ms) * 1000
        merge_sort_times[n] = t_ms
        inversion_counts[n] = count_ms

        # Benchmark Brute Force (feasible up to n=10,000; beyond that O(n^2) takes minutes)
        if n <= 10000:
            iters_bf = 10 if n <= 500 else (3 if n <= 2000 else 1)
            t_bf_start = time.perf_counter()
            for _ in range(iters_bf):
                count_bf = count_inversions_brute_force(arr)
            t_bf = ((time.perf_counter() - t_bf_start) / iters_bf) * 1000
            brute_force_times[n] = t_bf
            assert count_bf == count_ms, f"Count mismatch at n={n}: BF={count_bf}, MS={count_ms}"
            speedup = f"{t_bf / t_ms:.1f}x"
            bf_str = f"{t_bf:.4f}"
        else:
            # Extrapolate for comparison beyond 10,000 based on O(n^2) scaling
            t_bf_extrap = brute_force_times[10000] * ((n / 10000) ** 2)
            speedup = f"{t_bf_extrap / t_ms:.0f}x (est)"
            bf_str = f"{t_bf_extrap:.2f} (est)"

        print(f"{n:<10} | {count_ms:<14} | {bf_str:<18} | {t_ms:<16.4f} | {speedup:<10}")

    print("=" * 80)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    bf_sizes = sorted(brute_force_times.keys())
    bf_vals = [brute_force_times[s] for s in bf_sizes]

    ms_sizes = sorted(merge_sort_times.keys())
    ms_vals = [merge_sort_times[s] for s in ms_sizes]

    # Subplot 1: Linear Scale (up to n=10,000 where both ran empirically)
    ax1.plot(bf_sizes, bf_vals, marker="o", color="#d62728", linewidth=2.2, label="Brute Force O(n²)")
    ax1.plot(bf_sizes, [merge_sort_times[s] for s in bf_sizes], marker="s", color="#1f77b4", linewidth=2.2, label="Merge Sort O(n log n)")
    ax1.set_title("Execution Time vs. Input Size (Linear Scale, n ≤ 10,000)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Array Size (n)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=11)

    # Subplot 2: Log-Log Scale (full range n=10 up to 10^5)
    ax2.plot(bf_sizes, bf_vals, marker="o", color="#d62728", linewidth=2.2, label="Brute Force O(n²)")
    ax2.plot(ms_sizes, ms_vals, marker="s", color="#1f77b4", linewidth=2.2, label="Merge Sort O(n log n)")
    ax2.set_title("Execution Time vs. Input Size (Log-Log Scale, n = 10 to 10⁵)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Array Size (n) [Log Scale]", fontsize=11)
    ax2.set_ylabel("Execution Time (ms) [Log Scale]", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.legend(fontsize=11)

    plt.suptitle("Inversion Counting Performance Analysis: Brute Force vs. Divide and Conquer", fontsize=14, fontweight="bold")
    plt.tight_layout()

    for d in IMAGES_DIRS:
        save_path = d / "inversion_count_analysis.png"
        plt.savefig(save_path, dpi=300)
        print(f"[Plot Saved] {save_path}")

    if show_plot:
        plt.show()

    return brute_force_times, merge_sort_times


if __name__ == "__main__":
    benchmark_inversion_counting(show_plot=False)
