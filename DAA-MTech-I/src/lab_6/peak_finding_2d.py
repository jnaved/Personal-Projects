"""
Assignment 7:
Generalize the 1D peak finding problem to 2D.
A peak in a 2D matrix is an element greater than or equal to all its neighbours.
Your task is to find such a peak in a matrix.
Constraints: 1 <= n, m <= 10^5
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


def is_peak_2d(matrix: list[list[int]], r: int, c: int) -> bool:
    """Verifies whether matrix[r][c] is a peak relative to its 4 adjacent neighbors."""
    n = len(matrix)
    m = len(matrix[0])
    val = matrix[r][c]

    # Top
    if r > 0 and matrix[r - 1][c] > val:
        return False
    # Bottom
    if r < n - 1 and matrix[r + 1][c] > val:
        return False
    # Left
    if c > 0 and matrix[r][c - 1] > val:
        return False
    # Right
    if c < m - 1 and matrix[r][c + 1] > val:
        return False

    return True


def find_peak_2d_divide_and_conquer(matrix: list[list[int]]) -> tuple[int, int, int]:
    """
    Finds a 2D peak in O(n log m) time using Divide and Conquer.

    Algorithm:
      1. Pick middle column j = (c_start + c_end) // 2.
      2. Find row index i_max having the maximum element in column j (takes O(n)).
      3. Compare matrix[i_max][j] with its left and right neighbors:
         - If left neighbor > current: a peak is guaranteed in the left half; recurse left.
         - Else if right neighbor > current: a peak is guaranteed in the right half; recurse right.
         - Else: current element is >= left and right, and is >= top and bottom (global max in col j).
                 Thus, it is a 2D peak! Return (i_max, j, value).

    Recurrence: T(m) = T(m/2) + O(n) => T(n, m) = Theta(n log m).
    Space Complexity: O(1) iterative (or O(log m) recursive).
    """
    n = len(matrix)
    m = len(matrix[0])

    c_start = 0
    c_end = m - 1

    while c_start <= c_end:
        mid_c = c_start + (c_end - c_start) // 2

        # Step 2: Find global maximum of column mid_c
        max_r = 0
        max_val = matrix[0][mid_c]
        for r in range(1, n):
            if matrix[r][mid_c] > max_val:
                max_val = matrix[r][mid_c]
                max_r = r

        # Step 3: Check left and right horizontal neighbors
        has_greater_left = (mid_c > 0 and matrix[max_r][mid_c - 1] > max_val)
        has_greater_right = (mid_c < m - 1 and matrix[max_r][mid_c + 1] > max_val)

        if has_greater_left:
            # Recurse on left submatrix
            c_end = mid_c - 1
        elif has_greater_right:
            # Recurse on right submatrix
            c_start = mid_c + 1
        else:
            # Local peak found
            return max_r, mid_c, max_val

    # Fallback (should not be reached if inputs are non-empty)
    return 0, 0, matrix[0][0]


def find_peak_2d_brute_force(matrix: list[list[int]]) -> tuple[int, int, int]:
    """
    Brute-force approach: checks every cell until a peak is found.
    Time Complexity: O(n * m) worst case.
    """
    n = len(matrix)
    m = len(matrix[0])

    for i in range(n):
        for j in range(m):
            if is_peak_2d(matrix, i, j):
                return i, j, matrix[i][j]

    return 0, 0, matrix[0][0]


def benchmark_peak_finding_2d(show_plot: bool = True):
    """
    Compares Divide and Conquer O(n log m) with Brute Force O(n*m) on increasing matrix dimensions.
    """
    matrix_sizes = [50, 100, 200, 400, 800, 1500, 2500]

    dc_times = {}
    bf_times = {}

    rng = random.Random(42)

    print("=" * 75)
    print(f"{'Matrix Size (N x N)':<22} | {'DC Peak Found':<18} | {'DC Time (ms)':<16} | {'BF Time (ms)':<16}")
    print("=" * 75)

    for n in matrix_sizes:
        # Generate random N x N matrix
        matrix = [[rng.randint(1, 10_000_000) for _ in range(n)] for _ in range(n)]

        # Divide & Conquer Benchmark
        t0 = time.perf_counter()
        r_dc, c_dc, val_dc = find_peak_2d_divide_and_conquer(matrix)
        t_dc = (time.perf_counter() - t0) * 1000
        dc_times[n] = t_dc

        assert is_peak_2d(matrix, r_dc, c_dc), f"Invalid peak found by DC at ({r_dc}, {c_dc})"

        # Brute Force Benchmark
        if n <= 1500:
            t0 = time.perf_counter()
            r_bf, c_bf, val_bf = find_peak_2d_brute_force(matrix)
            t_bf = (time.perf_counter() - t0) * 1000
            bf_times[n] = t_bf
            assert is_peak_2d(matrix, r_bf, c_bf), f"Invalid peak found by BF at ({r_bf}, {c_bf})"
            bf_str = f"{t_bf:.4f}"
        else:
            bf_str = "N/A (O(N²))"

        print(f"{f'{n} x {n}':<22} | {f'({r_dc}, {c_dc})':<18} | {t_dc:<16.4f} | {bf_str:<16}")

    print("=" * 75)

    # Plot results
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ns = sorted(dc_times.keys())
    dc_vals = [dc_times[s] for s in ns]

    ax.plot(ns, dc_vals, marker="o", color="#1f77b4", linewidth=2.2, label="Divide & Conquer O(n log m)")

    bf_ns = sorted(bf_times.keys())
    bf_vals = [bf_times[s] for s in bf_ns]
    ax.plot(bf_ns, bf_vals, marker="s", color="#d62728", linewidth=2.2, label="Brute Force O(n · m)")

    ax.set_title("2D Peak Finding: Divide & Conquer vs. Brute Force Runtime", fontsize=12, fontweight="bold")
    ax.set_xlabel("Matrix Dimension N (for N x N matrix)", fontsize=11)
    ax.set_ylabel("Execution Time (ms)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()

    for d in IMAGES_DIRS:
        save_path = d / "peak_finding_2d_analysis.png"
        plt.savefig(save_path, dpi=300)
        print(f"[Plot Saved] {save_path}")

    if show_plot:
        plt.show()

    return dc_times, bf_times


if __name__ == "__main__":
    benchmark_peak_finding_2d(show_plot=False)
