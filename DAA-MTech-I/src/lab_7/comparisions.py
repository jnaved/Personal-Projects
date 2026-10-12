import time
import random
from pathlib import Path
import matplotlib.pyplot as plt
from naive_string_match import main as naive
from kmp import main as kmp
from rabin_karp import main as rk

CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR.parent.parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def generate_random_text(
    length: int | str,
    alphabet: str = "abcd",
    seed: int | None = None,
) -> str:
    """
    Generate a random text string of the specified length.

    :param length: Length of the generated text.
    :param alphabet: Characters to sample from (default: "abcd").
    :param seed: Optional random seed for reproducible results.
    :return: Randomly generated string.
    """
    rng = random.Random(seed) if seed is not None else random
    return "".join(rng.choices(alphabet, k=int(length)))


def generate_random_pattern(
    length: int | str,
    alphabet: str = "abcd",
    text: str | None = None,
    seed: int | None = None,
) -> str:
    """
    Generate a random pattern string of the specified length.
    If `text` is provided and long enough, extracts a random substring from it.
    Otherwise, generates a random pattern from `alphabet`.

    :param length: Length of the generated pattern.
    :param alphabet: Characters to sample from if text is not provided (default: "abcd").
    :param text: Optional source text to slice a guaranteed-matching pattern from.
    :param seed: Optional random seed for reproducible results.
    :return: Randomly generated pattern string.
    """
    rng = random.Random(seed) if seed is not None else random
    length = int(length)

    if text is not None and len(text) >= length:
        start_idx = rng.randint(0, len(text) - length)
        return text[start_idx : start_idx + length]

    return "".join(rng.choices(alphabet, k=length))


def count_naive_comparisons(text: str, pattern: str) -> int:
    """Count character comparisons performed by the Naive string matching algorithm."""
    n = len(text)
    m = len(pattern)
    comps = 0
    for i in range(n - m + 1):
        k = i
        for j in range(m):
            comps += 1
            if text[k] == pattern[j]:
                k += 1
            else:
                break
    return comps


def count_kmp_comparisons(text: str, pattern: str) -> int:
    """Count character comparisons performed by KMP (preprocessing + search)."""
    m = len(pattern)
    n = len(text)
    comps = 0

    # 1. LPS Preprocessing
    lps = [0] * m
    pattern_length = 0
    i = 1
    while i < m:
        comps += 1
        if pattern[i] == pattern[pattern_length]:
            pattern_length += 1
            lps[i] = pattern_length
            i += 1
        else:
            if pattern_length != 0:
                pattern_length = lps[pattern_length - 1]
            else:
                lps[i] = 0
                i += 1

    # 2. Search
    i, j = 0, 0
    while i < n:
        comps += 1
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return comps


def count_rk_comparisons(text: str, pattern: str, Q: int = 101) -> int:
    """Count character comparisons performed by Rabin-Karp (verification on hash match)."""
    n = len(text)
    m = len(pattern)
    h = pow(10, m - 1, Q)
    p_hash, t_hash = 0, 0
    comps = 0

    # Initial hash
    for i in range(m):
        p_hash = (10 * p_hash + ord(pattern[i])) % Q
        t_hash = (10 * t_hash + ord(text[i])) % Q

    for i in range(n - m + 1):
        if p_hash == t_hash:
            for j in range(m):
                comps += 1
                if text[i + j] != pattern[j]:
                    break

        if i < n - m:
            t_hash = (10 * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % Q
            if t_hash < 0:
                t_hash += Q

    return comps


# input_sizes >> pattern_sizes
input_sizes = [500, 1000, 10000]
pattern_sizes = [5, 10, 20]

n = len(input_sizes)

texts = [generate_random_text(length=i, seed=42 + idx) for idx, i in enumerate(input_sizes)]
patterns = [generate_random_pattern(length=pattern_sizes[i], text=texts[i], seed=100 + i) for i in range(n)]


def input_size_vs_execution_time(show_plot: bool = True):
    naive_input_size_time_map = {}
    kmp_input_size_time_map = {}
    rk_input_size_time_map = {}
    iterations = 5

    for i in range(n):
        t = time.perf_counter()
        for _ in range(iterations):
            naive(text=texts[i], pattern=patterns[i])
        naive_input_size_time_map[input_sizes[i]] = ((time.perf_counter() - t) / iterations) * 1000

        t = time.perf_counter()
        for _ in range(iterations):
            kmp(text=texts[i], pattern=patterns[i])
        kmp_input_size_time_map[input_sizes[i]] = ((time.perf_counter() - t) / iterations) * 1000

        t = time.perf_counter()
        for _ in range(iterations):
            rk(text=texts[i], pattern=patterns[i])
        rk_input_size_time_map[input_sizes[i]] = ((time.perf_counter() - t) / iterations) * 1000

    print("--- Input Size vs Execution Time (ms) ---")
    print(f"Naive:      {naive_input_size_time_map}")
    print(f"KMP:        {kmp_input_size_time_map}")
    print(f"Rabin-Karp: {rk_input_size_time_map}")

    plt.figure(figsize=(9, 5.5))
    plt.plot(input_sizes, [naive_input_size_time_map[s] for s in input_sizes], label="Naive String Match", marker="o", color="#d62728", linewidth=2)
    plt.plot(input_sizes, [kmp_input_size_time_map[s] for s in input_sizes], label="KMP", marker="s", color="#1f77b4", linewidth=2)
    plt.plot(input_sizes, [rk_input_size_time_map[s] for s in input_sizes], label="Rabin-Karp", marker="^", color="#2ca02c", linewidth=2)
    plt.title("Input Size vs. Execution Time", fontsize=13, fontweight="bold")
    plt.xlabel("Text Size (n)", fontsize=11)
    plt.ylabel("Execution Time (ms)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = IMAGES_DIR / "input_size_vs_execution_time.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Plot Saved] Figure saved to: {save_path}")

    if show_plot:
        plt.show()


def pattern_length_vs_execution_time(show_plot: bool = True):
    naive_pattern_size_time_map = {}
    kmp_pattern_size_time_map = {}
    rk_pattern_size_time_map = {}
    iterations = 5

    # Fix text length to isolate the effect of pattern length
    fixed_text = texts[-1]
    test_patterns = [generate_random_pattern(length=ps, text=fixed_text, seed=200 + idx) for idx, ps in enumerate(pattern_sizes)]

    for i in range(len(pattern_sizes)):
        ps = pattern_sizes[i]
        p = test_patterns[i]

        t = time.perf_counter()
        for _ in range(iterations):
            naive(text=fixed_text, pattern=p)
        naive_pattern_size_time_map[ps] = ((time.perf_counter() - t) / iterations) * 1000

        t = time.perf_counter()
        for _ in range(iterations):
            kmp(text=fixed_text, pattern=p)
        kmp_pattern_size_time_map[ps] = ((time.perf_counter() - t) / iterations) * 1000

        t = time.perf_counter()
        for _ in range(iterations):
            rk(text=fixed_text, pattern=p)
        rk_pattern_size_time_map[ps] = ((time.perf_counter() - t) / iterations) * 1000

    print("\n--- Pattern Length vs Execution Time (ms) ---")
    print(f"Naive:      {naive_pattern_size_time_map}")
    print(f"KMP:        {kmp_pattern_size_time_map}")
    print(f"Rabin-Karp: {rk_pattern_size_time_map}")

    plt.figure(figsize=(9, 5.5))
    plt.plot(pattern_sizes, [naive_pattern_size_time_map[p] for p in pattern_sizes], label="Naive String Match", marker="o", color="#d62728", linewidth=2)
    plt.plot(pattern_sizes, [kmp_pattern_size_time_map[p] for p in pattern_sizes], label="KMP", marker="s", color="#1f77b4", linewidth=2)
    plt.plot(pattern_sizes, [rk_pattern_size_time_map[p] for p in pattern_sizes], label="Rabin-Karp", marker="^", color="#2ca02c", linewidth=2)
    plt.title(f"Pattern Length vs. Execution Time (Text Size = {len(fixed_text)})", fontsize=13, fontweight="bold")
    plt.xlabel("Pattern Length (m)", fontsize=11)
    plt.ylabel("Execution Time (ms)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = IMAGES_DIR / "pattern_length_vs_execution_time.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Plot Saved] Figure saved to: {save_path}")

    if show_plot:
        plt.show()


def input_size_vs_comparisions(show_plot: bool = True):
    naive_comparisons_map = {}
    kmp_comparisons_map = {}
    rk_comparisons_map = {}

    for i in range(n):
        sz = input_sizes[i]
        naive_comparisons_map[sz] = count_naive_comparisons(texts[i], patterns[i])
        kmp_comparisons_map[sz] = count_kmp_comparisons(texts[i], patterns[i])
        rk_comparisons_map[sz] = count_rk_comparisons(texts[i], patterns[i])

    print("\n--- Input Size vs Character Comparisons ---")
    print(f"Naive:      {naive_comparisons_map}")
    print(f"KMP:        {kmp_comparisons_map}")
    print(f"Rabin-Karp: {rk_comparisons_map}")

    plt.figure(figsize=(9, 5.5))
    plt.plot(input_sizes, [naive_comparisons_map[s] for s in input_sizes], label="Naive String Match", marker="o", color="#d62728", linewidth=2)
    plt.plot(input_sizes, [kmp_comparisons_map[s] for s in input_sizes], label="KMP", marker="s", color="#1f77b4", linewidth=2)
    plt.plot(input_sizes, [rk_comparisons_map[s] for s in input_sizes], label="Rabin-Karp (Verification Comps)", marker="^", color="#2ca02c", linewidth=2)
    plt.title("Input Size vs. Number of Comparisons", fontsize=13, fontweight="bold")
    plt.xlabel("Text Size (n)", fontsize=11)
    plt.ylabel("Number of Character Comparisons", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = IMAGES_DIR / "input_size_vs_comparisons.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Plot Saved] Figure saved to: {save_path}")

    if show_plot:
        plt.show()


# Alias with standard spelling
input_size_vs_comparisons = input_size_vs_comparisions


if __name__ == "__main__":
    input_size_vs_execution_time()
    pattern_length_vs_execution_time()
    input_size_vs_comparisions()