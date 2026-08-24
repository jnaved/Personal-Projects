'''
Implement linear search and binary search for array sizes of 10, 50, 100, 200
generate plots of input-size vs execution_time and compare with theoretical times
'''

import sys
import time
import random
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_FILE_PATH = CURRENT_DIR / "sample_input_1.txt"
DEFAULT_SIZES = [10, 50, 100, 200]


def generate_sample_input_file(filepath: str | Path = DEFAULT_FILE_PATH, target: int = 42, sizes: list[int] = DEFAULT_SIZES) -> str:
    """
    Generates a sample input file in the requested format:
    - First element: target value
    - Size 1 (e.g. 10)
    - 10 elements
    - Size 2 (e.g. 50)
    - 50 elements
    - Size 3 (e.g. 100)
    - 100 elements
    - Size 4 (e.g. 200)
    - 200 elements
    """
    filepath = Path(filepath)
    if not filepath.is_absolute():
        filepath = CURRENT_DIR / filepath

    lines = []
    # 1. First element is target value
    lines.append(f"# Target Value\n{target}")

    # 2. For each size, write size followed by space-separated elements
    for sz in sizes:
        # Generate sorted unique random numbers
        elements = sorted(random.sample(range(1, max(1000, sz * 20)), sz))
        lines.append(f"# Size: {sz}\n{sz}")
        lines.append(" ".join(map(str, elements)))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Sample input file generated at: {filepath}")
    return str(filepath)


def read_input_from_file(filepath: str | Path = DEFAULT_FILE_PATH) -> tuple[int, dict[int, list[int]]]:
    """
    Reads the target value and the multi-sized datasets from the input file.
    Returns:
        (target_value, {size: [elements, ...], ...})
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

    target_val = tokens[0]
    idx = 1
    datasets: dict[int, list[int]] = {}

    while idx < len(tokens):
        sz = tokens[idx]
        idx += 1
        arr = tokens[idx: idx + sz]
        datasets[sz] = arr
        idx += sz

    return target_val, datasets


def linear_search(target: int, array: list[int]) -> int:
    for idx, ele in enumerate(array):
        if ele == target:
            return idx
    return -1


def binary_search(target: int, array: list[int]) -> int:
    low = 0
    high = len(array) - 1

    while low <= high:
        mid = low + (high - low) // 2

        if array[mid] == target:
            return mid
        elif array[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


def implement_search(type: str, target: int, array: list[int]) -> int:
    if type == "linear":
        return linear_search(target=target, array=array)
    elif type == "binary":
        return binary_search(target=target, array=array)
    return -1


def get_target_for_case(case: str, array: list[int]) -> int:
    """
    Returns the target value for each of the 4 test cases:
    - 'beginning': First element (Best case for linear search)
    - 'middle': Middle element (Best case for binary search)
    - 'end': Last element (Worst case for linear search)
    - 'not_present': Element not in array (Worst case for both)
    """
    if case == "beginning":
        return array[0]
    elif case == "middle":
        l = len(array)
        return array[l // 2] if l&1 else array[(l-1)//2]
    elif case == "end":
        return array[-1]
    elif case == "not_present":
        return max(array) + 1000
    else:
        raise ValueError(f"Unknown case: {case}")


def measure_search_time(search_type: str, target: int, array: list[int], iterations: int = 10000) -> float:
    """
    Measures the average execution time in microseconds (µs) over multiple iterations.
    """
    search_fn = linear_search if search_type == "linear" else binary_search
    
    start = time.perf_counter()
    for _ in range(iterations):
        search_fn(target, array)
    total_time = time.perf_counter() - start

    # Return average execution time in microseconds (µs)
    return (total_time / iterations) * 1e6


def plot_and_compare_TC(type: str, datasets: dict[int, list[int]] | None = None):
    """
    Plots input-size vs execution time for 4 target cases:
    beginning, middle, end, not_present.
    """
    if datasets is None:
        _, datasets = read_input_from_file()

    sizes = sorted(datasets.keys())
    cases = ["beginning", "middle", "end", "not_present"]
    case_styles = {
        "beginning": {"color": "#2ca02c", "marker": "o", "label": "Beginning"},
        "middle": {"color": "#1f77b4", "marker": "s", "label": "Middle"},
        "end": {"color": "#ff7f0e", "marker": "^", "label": "End"},
        "not_present": {"color": "#d62728", "marker": "x", "label": "Not Present (Worst)"}
    }

    results: dict[str, list[float]] = {case: [] for case in cases}

    print(f"\n--- Benchmark Results for {type.capitalize()} Search (in µs) ---")
    print(f"{'Size (n)':<10} | " + " | ".join(f"{c:<14}" for c in cases))
    print("-" * 75)

    for sz in sizes:
        arr = datasets[sz]
        row_str = f"{sz:<10} | "
        for case in cases:
            target = get_target_for_case(case, arr)
            avg_time_us = measure_search_time(type, target, arr)
            results[case].append(avg_time_us)
            row_str += f"{avg_time_us:<14.4f} | "
        print(row_str)

    # Plotting
    plt.figure(figsize=(9, 6))
    for case in cases:
        plt.plot(
            sizes,
            results[case],
            marker=case_styles[case]["marker"],
            color=case_styles[case]["color"],
            linewidth=2,
            markersize=7,
            label=case_styles[case]["label"]
        )

    algo_title = "Linear Search" if type == "linear" else "Binary Search"
    plt.title(f"{algo_title}: Input Size vs Execution Time Across 4 Cases", fontsize=13, fontweight="bold")
    plt.xlabel("Input Size (n)", fontsize=11)
    plt.ylabel("Execution Time (µs)", fontsize=11)
    plt.xticks(sizes)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    # Save figure to images folder if available, else current directory
    images_dir = CURRENT_DIR.parent.parent / "images"
    save_path = images_dir / f"{type}_search_tc_analysis.png" if images_dir.exists() else CURRENT_DIR / f"{type}_search_tc_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Plot Saved] Saved figure to: {save_path}")

    # Also sync with overleaf/lab_2/images if present
    overleaf_images_dir = CURRENT_DIR.parent.parent / "overleaf" / "lab_2" / "images"
    if overleaf_images_dir.exists():
        plt.savefig(overleaf_images_dir / f"{type}_search_tc_analysis.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and any(arg.lower().lstrip("-") == "new" for arg in sys.argv[1:]):
        generate_sample_input_file()
    
    target, datasets = read_input_from_file()
    
    # Run searches on sample dataset
    sample_array = datasets[DEFAULT_SIZES[0]]
    print(f"Dataset sizes available: {list(datasets.keys())}")
    print(f"Target value from file: {target}")
    print(f"Linear search result on size {len(sample_array)}: index = {implement_search('linear', target, sample_array)}")
    print(f"Binary search result on size {len(sample_array)}: index = {implement_search('binary', target, sample_array)}")

    # Plot and compare for both linear and binary search
    plot_and_compare_TC(type="linear", datasets=datasets)
    plot_and_compare_TC(type="binary", datasets=datasets)