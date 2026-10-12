# Find element at index i if the array were sorted (k-th order statistic)
import heapq
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt


def sorting_selection(arr: list[int], target_index: int) -> int:
    return sorted(arr)[target_index]


def priority_queue_selection(arr: list[int], target_index: int) -> int:
    min_heap = arr.copy()
    heapq.heapify(min_heap)
    for _ in range(target_index):
        heapq.heappop(min_heap)
    return min_heap[0]


def partition(arr: list[int], low: int, high: int) -> int:
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    pivot_value = arr[high]
    store_index = low

    for scan_index in range(low, high):
        if arr[scan_index] <= pivot_value:
            arr[store_index], arr[scan_index] = arr[scan_index], arr[store_index]
            store_index += 1

    arr[store_index], arr[high] = arr[high], arr[store_index]
    return store_index


def quickselect(arr: list[int], target_index: int) -> int:
    elements = arr.copy()
    low = 0
    high = len(elements) - 1

    while low <= high:
        pivot_pos = partition(elements, low, high)
        if pivot_pos == target_index:
            return elements[pivot_pos]
        elif pivot_pos < target_index:
            low = pivot_pos + 1
        else:
            high = pivot_pos - 1

    return elements[target_index]


def generate_random_array(size: int, seed: int = 42) -> list[int]:
    random.seed(seed + size)
    return [random.randint(1, 100000) for _ in range(size)]


def run_static(sizes: list[int] = None):
    if sizes is None:
        sizes = [10, 50, 100, 200, 500, 1000]

    sort_times = []
    pq_times = []
    qs_times = []

    print("\n" + "=" * 96)
    print("      K-TH ELEMENT SELECTION: FULL SORT vs. PRIORITY QUEUE vs. QUICKSELECT")
    print("=" * 96)
    print(f"{'Size (N)':<10} | {'Index (i)':<10} | {'Full Sort (ms)':<16} | {'Priority Queue (ms)':<20} | {'Quickselect (ms)':<18} | {'Match'}")
    print("-" * 96)

    trials = 20
    for sz in sizes:
        target_idx = sz // 2
        test_data = generate_random_array(size=sz)

        # 1. Full Sort Benchmark
        t0 = time.perf_counter()
        for _ in range(trials):
            sort_res = sorting_selection(test_data, target_idx)
        sort_time = ((time.perf_counter() - t0) / trials) * 1000

        # 2. Priority Queue Benchmark
        t1 = time.perf_counter()
        for _ in range(trials):
            pq_res = priority_queue_selection(test_data, target_idx)
        pq_time = ((time.perf_counter() - t1) / trials) * 1000

        # 3. Quickselect Benchmark
        t2 = time.perf_counter()
        for _ in range(trials):
            qs_res = quickselect(test_data, target_idx)
        qs_time = ((time.perf_counter() - t2) / trials) * 1000

        sort_times.append(sort_time)
        pq_times.append(pq_time)
        qs_times.append(qs_time)

        is_match = (sort_res == pq_res == qs_res)
        match_str = "Yes" if is_match else "No"
        print(f"{sz:<10} | {target_idx:<10} | {sort_time:<16.4f} | {pq_time:<20.4f} | {qs_time:<18.4f} | {match_str}")

    print("=" * 96)

    # Demonstrate Priority Queue index sensitivity (near 0 vs near n)
    print("\n[Priority Queue Complexity vs. Target Index (N = 1000)]")
    large_n = 1000
    sample_data = generate_random_array(size=large_n)
    for test_idx in [0, large_n // 4, large_n // 2, (3 * large_n) // 4, large_n - 1]:
        t_start = time.perf_counter()
        for _ in range(trials):
            priority_queue_selection(sample_data, test_idx)
        elapsed = ((time.perf_counter() - t_start) / trials) * 1000
        print(f"  Target Index i = {test_idx:<4} ({test_idx / (large_n - 1) * 100:5.1f}% of N) -> Time: {elapsed:.4f} ms")

    # Plotting Benchmark Comparison
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Linear Scale
        ax1.plot(sizes, sort_times, label="Full Sorting $O(n \\log n)$", color="#ff7f0e", marker="s", linewidth=2.2, markersize=6)
        ax1.plot(sizes, pq_times, label="Priority Queue $O(n + i \\log n)$", color="#2ca02c", marker="^", linewidth=2.2, markersize=6)
        ax1.plot(sizes, qs_times, label="Quickselect $O(n)$", color="#1f77b4", marker="o", linewidth=2.2, markersize=6)
        ax1.set_title("Execution Time vs. Input Size (Linear Scale)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Array Size (N)", fontsize=11)
        ax1.set_ylabel("Execution Time (ms)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(fontsize=10)

        # Subplot 2: Log-Log Scale
        ax2.plot(sizes, sort_times, label="Full Sorting $O(n \\log n)$", color="#ff7f0e", marker="s", linewidth=2.2, markersize=6)
        ax2.plot(sizes, pq_times, label="Priority Queue $O(n + i \\log n)$", color="#2ca02c", marker="^", linewidth=2.2, markersize=6)
        ax2.plot(sizes, qs_times, label="Quickselect $O(n)$", color="#1f77b4", marker="o", linewidth=2.2, markersize=6)
        ax2.set_title("Execution Time vs. Input Size (Log-Log Scale)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Array Size (N) [Log Scale]", fontsize=11)
        ax2.set_ylabel("Execution Time (ms) [Log Scale]", fontsize=11)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", linestyle="--", alpha=0.6)
        ax2.legend(fontsize=10)

        plt.suptitle("K-th Order Statistic Selection: Time Complexity Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save plot to images directory
        current_dir = Path(__file__).resolve().parent
        images_dir = current_dir.parent.parent / "images"
        save_path = images_dir / "kth_element_tc_analysis.png" if images_dir.exists() else current_dir / "kth_element_tc_analysis.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n[Plot Saved] Figure saved to: {save_path}")

        # Sync with overleaf if present
        overleaf_images_dir = current_dir.parent.parent / "overleaf" / "lab_5" / "images"
        if overleaf_images_dir.exists():
            plt.savefig(overleaf_images_dir / "kth_element_tc_analysis.png", dpi=300)

        plt.show()
    except Exception as e:
        print(f"\n[Plotting Skipped or Display Unavailable]: {e}")


def find_kth_element():
    print("Enter the array size to execute: default(0 for static benchmark)")
    n = int(input().strip() or 0)
    if not n:
        run_static()
    else:
        arr = list(map(int, input(f"Enter {n} space-separated integers: ").split()))
        target_idx = int(input(f"Enter target index (0 to {n - 1}): ").strip())
        res = quickselect(arr, target_idx)
        print(f"Element at index {target_idx} when sorted: {res}")


find_kth_element()
