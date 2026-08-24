import random
import sys
import time
import matplotlib.pyplot as plt
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_FILE_PATH = CURRENT_DIR / "sample_input_3.txt"
DEFAULT_ROW_SIZES = [10, 50, 100, 200, 500]


def generate_sample_input_file(filepath: str | Path = DEFAULT_FILE_PATH, sizes: list[int] = DEFAULT_ROW_SIZES) -> str:
    """
    Generates a sample input file containing two matrices (Matrix A and Matrix B)
    for each specified dimension size (e.g. 10x10, 50x50, 100x100, 200x200, 500x500).
    """
    filepath = Path(filepath)
    if not filepath.is_absolute():
        filepath = CURRENT_DIR / filepath

    lines = []

    for row_size in sizes:
        lines.append(f"# Size: {row_size}\n{row_size}")
        
        # Generate Matrix A (row_size x row_size)
        lines.append(f"# Matrix A ({row_size}x{row_size})")
        for _ in range(row_size):
            row = [random.randint(1, 100) for _ in range(row_size)]
            lines.append(" ".join(map(str, row)))
            
        # Generate Matrix B (row_size x row_size)
        lines.append(f"# Matrix B ({row_size}x{row_size})")
        for _ in range(row_size):
            row = [random.randint(1, 100) for _ in range(row_size)]
            lines.append(" ".join(map(str, row)))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Sample input file generated at: {filepath}")
    return str(filepath)


def read_input_from_file(filepath: str | Path = DEFAULT_FILE_PATH) -> dict[int, tuple[list[list[int]], list[list[int]]]]:
    """
    Reads the two matrices (A, B) for each dimension size from the input file.
    Returns:
        {size: (matrix_A, matrix_B), ...}
    """
    filepath = Path(filepath)
    if not filepath.is_absolute():
        filepath = CURRENT_DIR / filepath

    if not filepath.exists():
        generate_sample_input_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not raw_lines:
        raise ValueError(f"Input file {filepath} is empty or corrupted.")

    tokens: list[int] = []
    for line in raw_lines:
        tokens.extend([int(x) for x in line.replace(",", " ").split() if x.strip()])

    if not tokens:
        raise ValueError("No numeric data found in input file.")

    idx = 0
    datasets: dict[int, tuple[list[list[int]], list[list[int]]]] = {}

    while idx < len(tokens):
        sz = tokens[idx]
        idx += 1
        
        # Read Matrix A
        mat_A = []
        for _ in range(sz):
            mat_A.append(tokens[idx : idx + sz])
            idx += sz
            
        # Read Matrix B
        mat_B = []
        for _ in range(sz):
            mat_B.append(tokens[idx : idx + sz])
            idx += sz

        datasets[sz] = (mat_A, mat_B)

    return datasets


def matrix_addition(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """
    Performs element-wise matrix addition: C = A + B
    Time Complexity: O(n^2)
    """
    n = len(A)
    C = [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]
    return C


def matrix_multiplication(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """
    Performs matrix multiplication: C = A * B
    Time Complexity: O(n^3)
    """
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matrix_transpose(A: list[list[int]]) -> list[list[int]]:
    """
    Computes transpose of matrix: A^T
    Time Complexity: O(n^2)
    """
    n = len(A)
    C = [[A[j][i] for j in range(n)] for i in range(n)]
    return C


def measure_operation_time(op_type: str, mat_A: list[list[int]], mat_B: list[list[int]]) -> float:
    """
    Measures the average execution time in microseconds (µs).
    Adjusts iteration count based on matrix order for accurate and fast benchmarking.
    """
    n = len(mat_A)
    
    # Adaptive iterations based on matrix size to balance accuracy & performance
    if n <= 10:
        iterations = 100
    elif n <= 50:
        iterations = 15
    elif n <= 100:
        iterations = 5
    elif n <= 200:
        iterations = 2
    else:
        iterations = 1

    if op_type == "add":
        start = time.perf_counter()
        for _ in range(iterations):
            matrix_addition(mat_A, mat_B)
        total_time = time.perf_counter() - start
    elif op_type == "multiply":
        start = time.perf_counter()
        for _ in range(iterations):
            matrix_multiplication(mat_A, mat_B)
        total_time = time.perf_counter() - start
    elif op_type == "transpose":
        start = time.perf_counter()
        for _ in range(iterations):
            matrix_transpose(mat_A)
        total_time = time.perf_counter() - start
    else:
        raise ValueError(f"Unknown operation type: {op_type}")

    # Average execution time in microseconds (µs)
    return (total_time / iterations) * 1e6


def plot_and_compare_matrix_operations(datasets: dict[int, tuple[list[list[int]], list[list[int]]]] | None = None):
    """
    Benchmarks Matrix Addition, Multiplication, and Transpose across all matrix orders
    and plots all 3 operations in 1 comparative graph.
    """
    if datasets is None:
        datasets = read_input_from_file()

    sizes = sorted(datasets.keys())
    operations = ["add", "multiply", "transpose"]
    op_styles = {
        "add": {"color": "#1f77b4", "marker": "o", "label": "Matrix Addition (O(n²))"},
        "multiply": {"color": "#d62728", "marker": "s", "label": "Matrix Multiplication (O(n³))"},
        "transpose": {"color": "#2ca02c", "marker": "^", "label": "Matrix Transpose (O(n²))"}
    }

    results: dict[str, list[float]] = {op: [] for op in operations}

    print("\n--- Benchmark Results for Matrix Operations (in µs) ---")
    print(f"{'Order (n x n)':<15} | " + " | ".join(f"{op.capitalize():<18}" for op in operations))
    print("-" * 75)

    for sz in sizes:
        mat_A, mat_B = datasets[sz]
        row_str = f"{f'{sz}x{sz}':<15} | "
        for op in operations:
            avg_time_us = measure_operation_time(op, mat_A, mat_B)
            results[op].append(avg_time_us)
            row_str += f"{avg_time_us:<18.2f} | "
        print(row_str)

    # Plot all 3 operations in a single graph
    plt.figure(figsize=(10, 6))
    for op in operations:
        plt.plot(
            sizes,
            results[op],
            marker=op_styles[op]["marker"],
            color=op_styles[op]["color"],
            linewidth=2,
            markersize=7,
            label=op_styles[op]["label"]
        )

    plt.title("Matrix Operations: Dimension Size (n) vs Execution Time", fontsize=13, fontweight="bold")
    plt.xlabel("Matrix Dimension Size (n for n x n matrix)", fontsize=11)
    plt.ylabel("Execution Time (µs)", fontsize=11)
    plt.xticks(sizes, [f"{s}x{s}" for s in sizes])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    # Save figure to images folder if available, else current directory
    images_dir = CURRENT_DIR.parent.parent / "images"
    save_path = images_dir / "matrix_operations_tc_analysis.png" if images_dir.exists() else CURRENT_DIR / "matrix_operations_tc_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Plot Saved] Saved figure to: {save_path}")

    # Also sync with overleaf/lab_2/images if present
    overleaf_images_dir = CURRENT_DIR.parent.parent / "overleaf" / "lab_2" / "images"
    if overleaf_images_dir.exists():
        plt.savefig(overleaf_images_dir / "matrix_operations_tc_analysis.png", dpi=300)


if __name__ == "__main__":
    if len(sys.argv) > 1 and any(arg.lower().lstrip("-") == "new" for arg in sys.argv[1:]):
        generate_sample_input_file()

    datasets = read_input_from_file()
    plot_and_compare_matrix_operations(datasets=datasets)