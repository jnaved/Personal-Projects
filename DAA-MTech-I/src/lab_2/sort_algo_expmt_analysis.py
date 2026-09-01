import random
import sys
import time
import matplotlib.pyplot as plt
from pathlib import Path



CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_FILE_PATH = CURRENT_DIR / "sample_input_2.txt"
DEFAULT_SIZES = [10, 50, 100, 200]


def generate_sample_input_file(filepath: str | Path = DEFAULT_FILE_PATH, sizes: list[int] = DEFAULT_SIZES) -> str:
    """
    Generates a sample input file in the requested format:
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

    # 2. For each size, write size followed by space-separated elements
    for sz in sizes:
        # Generate unique random numbers
        elements = random.sample(range(1, max(1000, sz * 20)), sz)
        lines.append(f"# Size: {sz}\n{sz}")
        lines.append(" ".join(map(str, elements)))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Sample input file generated at: {filepath}")
    return str(filepath)


def read_input_from_file(filepath: str | Path = DEFAULT_FILE_PATH) -> dict[int, list[int]]:
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
    
    idx = 0
    datasets: dict[int, list[int]] = {}

    while idx < len(tokens):
        sz = tokens[idx]
        idx += 1
        arr = tokens[idx: idx + sz]
        datasets[sz] = arr
        idx += sz

    return datasets


def bubble_sort(array: list[int]) -> list[int]:
    """
        Implements bubble sort
    """
    n = len(array)
    for i in range(n):
        for j in range(0, n-i-1):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]


def selection_sort(array: list[int]):
    """
        Implements selection sort
    """
    n = len(array)
    for i in range(n-1):
        min_idx = i
        for j in range(i+1, n):
            if array[j] < array[min_idx]:
                min_idx = j
        
        array[i], array[min_idx] = array[min_idx], array[i]


def insertion_sort(array: list[int]):
    """
        Implements insertion sort
    """
    for i in range(1, len(array)):
        key = array[i]
        j = i-1
        while(j>=0 and key < array[j]):
            array[j+1] = array[j]
            j-=1
        array[j+1] = key


def get_target_for_case(case: str, array: list[int]) -> list[int]:
    """
    Returns the target value for each of the 4 test cases:
    - 'random': Returns the array
    - 'sorted': Return sorted array
    - 'rsorted': Return reverse sorted array
    """
    if case == "random":
        return array
    elif case == "sorted":
        return sorted(array)
    elif case == "rsorted":
        return sorted(array, reverse=True)
    else:
        raise ValueError(f"Unknown case: {case}")


def measure_sort_time(sort_type: str, array: list[int], iterations: int = 10000) -> float:
    """
    Measures the average execution time in microseconds (µs) over multiple iterations.
    """
    sort_fn_map = {
        "bubble": bubble_sort,
        "selection": selection_sort,
        "insertion": insertion_sort
    }
    sort_fn = sort_fn_map[sort_type]
    
    start = time.perf_counter()
    for _ in range(iterations):
        sort_fn(array.copy())
    total_time = time.perf_counter() - start

    # Return average execution time in microseconds (µs)
    return (total_time / iterations) * 1e6


def plot_and_compare_TC(type: str, datasets: dict[int, list[int]] | None = None):
    """
    Plots input-size vs execution time for 4 target cases:
    beginning, middle, end, not_present.
    """
    if datasets is None:
        datasets = read_input_from_file()

    sizes = sorted(datasets.keys())
    cases = ["random", "sorted", "rsorted"]
    case_styles = {
        "random": {"color": "#2ca02c", "marker": "o", "label": "Random"},
        "sorted": {"color": "#1f77b4", "marker": "s", "label": "Sorted"},
        "rsorted": {"color": "#ff7f0e", "marker": "^", "label": "Reversed Sorted"}
    }

    results: dict[str, list[float]] = {case: [] for case in cases}

    print(f"\n--- Benchmark Results for {type.capitalize()} Sort (in µs) ---")
    print(f"{'Size (n)':<10} | " + " | ".join(f"{c:<14}" for c in cases))
    print("-" * 75)

    for sz in sizes:
        arr = datasets[sz]
        row_str = f"{sz:<10} | "
        for case in cases:
            case_arr = get_target_for_case(case, arr)
            avg_time_us = measure_sort_time(type, case_arr)
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

    algo_title_map = {
        "bubble": "Bubble Sort",
        "selection": "Selection Sort",
        "insertion": "Insertion Sort"
    }
    algo_title = algo_title_map[type]
    plt.title(f"{algo_title}: Input Size vs Execution Time", fontsize=13, fontweight="bold")
    plt.xlabel("Input Size (n)", fontsize=11)
    plt.ylabel("Execution Time (µs)", fontsize=11)
    plt.xticks(sizes)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    # Save figure to images folder if available, else current directory
    images_dir = CURRENT_DIR.parent.parent / "images"
    save_path = images_dir / f"{type}_sort_tc_analysis.png" if images_dir.exists() else CURRENT_DIR / f"{type}_sort_tc_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Plot Saved] Saved figure to: {save_path}")

    # Also sync with overleaf/lab_2/images if present
    overleaf_images_dir = CURRENT_DIR.parent.parent / "overleaf" / "lab_2" / "images"
    if overleaf_images_dir.exists():
        plt.savefig(overleaf_images_dir / f"{type}_sort_tc_analysis.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and any(arg.lower().lstrip("-") == "new" for arg in sys.argv[1:]):
        generate_sample_input_file()
    
    for type in ("bubble", "selection", "insertion"):
        plot_and_compare_TC(type=type)