import math
import time
from pathlib import Path
import matplotlib.pyplot as plt


def code1(n: int) -> int:
    """
    (a) Code 1:
        for (i = 1; i <= n; i++)
            for (j = 1; j <= i; j++)
                x = x + 1;
    Total operations: sum_{i=1}^n i = n*(n + 1)/2 = Theta(n^2)
    """
    x = 0
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            x += 1
    return x


def code2(n: int) -> int:
    """
    (b) Code 2:
        j = n
        while (j >= 1) {
            for (i = 1; i <= j; i++)
                x = x + 1;
            j = j / 2;  # Intended algorithm (halving outer loop)
        }
    Total operations: sum_{k=0}^{floor(log2 n)} floor(n / 2^k) = Theta(n)
    """
    x = 0
    j = n

    while j >= 1:
        for i in range(1, j + 1):
            x += 1
        j = j // 2
    return x


def code3(n: int) -> int:
    """
    (c) Code 3:
        for (i = 1; i <= n; i++)
            for (j = 1; j <= i; j++)
                for (k = 1; k <= i; k++)
                    x = x + 1;
    Notice: both j and k loop from 1 up to i.
    Total operations: sum_{i=1}^n (i * i) = sum_{i=1}^n i^2
                    = n*(n + 1)*(2n + 1)/6 = Theta(n^3)
    """
    x = 0
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            for k in range(1, i + 1):
                x += 1
    return x


def code4(n: int) -> int:
    """
    (d) Code 4:
        i = n
        while (i >= 1) {
            for (j = 1; j <= n; j++)
                x = x + 1;
            i = i / 2;
        }
    Outer loop runs: floor(log2 n) + 1 times.
    Inner loop runs: n times per outer iteration.
    Total operations: (floor(log2 n) + 1) * n = Theta(n log n)
    """
    x = 0
    i = n
    while i >= 1:
        for j in range(1, n + 1):
            x += 1
        i = i // 2
    return x


def theoretical_count_code1(n: int) -> int:
    """Exact count: n * (n + 1) // 2 ~ (1/2) * n^2"""
    return n * (n + 1) // 2


def theoretical_count_code2(n: int) -> int:
    """Exact count: sum_{k=0}^{floor(log2 n)} floor(n / 2^k) ~ 2*n"""
    total = 0
    curr = n
    while curr >= 1:
        total += curr
        curr //= 2
    return total


def theoretical_count_code3(n: int) -> int:
    """Exact count: n * (n + 1) * (2*n + 1) // 6 ~ (1/3) * n^3"""
    return n * (n + 1) * (2 * n + 1) // 6


def theoretical_count_code4(n: int) -> int:
    """Exact count: (floor(log2 n) + 1) * n ~ n * log2(n)"""
    if n <= 0:
        return 0
    k = n.bit_length()  # floor(log2(n)) + 1
    return k * n


# =====================================================================
# 3. Timing & Benchmarking Suite
# =====================================================================

def measure_execution_time(func, n: int, trials: int = 5) -> tuple[int, float]:
    """
    Executes func(n) across multiple trials using time.perf_counter().
    Returns the operation count and the average execution time in milliseconds.
    """
    # Execute once to get accurate operation count
    op_count = func(n)

    # Benchmark average wall-clock execution time
    t_start = time.perf_counter()
    for _ in range(trials):
        func(n)
    elapsed_ms = ((time.perf_counter() - t_start) / trials) * 1000.0

    return op_count, elapsed_ms


def run_static(sizes: list[int] = None):
    """
    Runs automated benchmarks across varying input sizes n, comparing empirical
    cost against theoretical closed-forms and asymptotic bounds Theta(g(n)).
    """
    if sizes is None:
        # Sizes tailored to test all regimes up to cubic complexity without excessive waiting
        sizes = [10, 25, 50, 100, 150, 200, 300, 400]

    # Data collection dictionaries
    empirical_times = {"Code 1": [], "Code 2": [], "Code 3": [], "Code 4": []}
    empirical_counts = {"Code 1": [], "Code 2": [], "Code 3": [], "Code 4": []}
    theoretical_counts = {"Code 1": [], "Code 2": [], "Code 3": [], "Code 4": []}

    print("\n" + "=" * 115)
    print("                EMPIRICAL TIMING & ASYMPTOTIC COMPLEXITY ANALYSIS (ASSIGNMENT 5)")
    print("=" * 115)
    print(
        "Code 1: Theta(n^2)    | Code 2: Theta(n)    | Code 3: Theta(n^3)    | Code 4: Theta(n log n)"
    )
    print("=" * 115)

    header = (
        f"{'N':<6} | "
        f"{'Code 1 (ms)':<12} {'x_1 count':<11} | "
        f"{'Code 2 (ms)':<12} {'x_2 count':<11} | "
        f"{'Code 3 (ms)':<12} {'x_3 count':<12} | "
        f"{'Code 4 (ms)':<12} {'x_4 count':<11} | Match"
    )
    print(header)
    print("-" * 115)

    trials = 5
    for n in sizes:
        # Measure Code 1: Theta(n^2)
        c1, t1 = measure_execution_time(code1, n, trials=trials)
        exp_c1 = theoretical_count_code1(n)

        # Measure Code 2: Theta(n)
        c2, t2 = measure_execution_time(code2, n, trials=trials)
        exp_c2 = theoretical_count_code2(n)

        # Measure Code 3: Theta(n^3) (reduce trials for larger n if needed)
        c3_trials = 3 if n >= 300 else trials
        c3, t3 = measure_execution_time(code3, n, trials=c3_trials)
        exp_c3 = theoretical_count_code3(n)

        # Measure Code 4: Theta(n log n)
        c4, t4 = measure_execution_time(code4, n, trials=trials)
        exp_c4 = theoretical_count_code4(n)

        # Record metrics
        empirical_times["Code 1"].append(t1)
        empirical_times["Code 2"].append(t2)
        empirical_times["Code 3"].append(t3)
        empirical_times["Code 4"].append(t4)

        empirical_counts["Code 1"].append(c1)
        empirical_counts["Code 2"].append(c2)
        empirical_counts["Code 3"].append(c3)
        empirical_counts["Code 4"].append(c4)

        theoretical_counts["Code 1"].append(exp_c1)
        theoretical_counts["Code 2"].append(exp_c2)
        theoretical_counts["Code 3"].append(exp_c3)
        theoretical_counts["Code 4"].append(exp_c4)

        all_matched = (c1 == exp_c1 and c2 == exp_c2 and c3 == exp_c3 and c4 == exp_c4)
        match_str = "Yes" if all_matched else "No"

        row = (
            f"{n:<6} | "
            f"{t1:<12.4f} {c1:<11} | "
            f"{t2:<12.4f} {c2:<11} | "
            f"{t3:<12.4f} {c3:<12} | "
            f"{t4:<12.4f} {c4:<11} | {match_str}"
        )
        print(row)

    print("=" * 115)

    try:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

        colors = {
            "Code 1": "#ff7f0e",  # Orange
            "Code 2": "#2ca02c",  # Green
            "Code 3": "#d62728",  # Red
            "Code 4": "#1f77b4",  # Blue
        }
        markers = {
            "Code 1": "s",
            "Code 2": "^",
            "Code 3": "D",
            "Code 4": "o",
        }

        # -------------------------------------------------------------
        # Subplot 1: Empirical Execution Time (Linear Scale)
        # -------------------------------------------------------------
        for code_name in empirical_times:
            ax1.plot(
                sizes,
                empirical_times[code_name],
                label=f"{code_name}",
                color=colors[code_name],
                marker=markers[code_name],
                linewidth=2.2,
                markersize=6,
            )
        ax1.set_title("Execution Time vs. Input Size (Linear Scale)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Input Size (n)", fontsize=11)
        ax1.set_ylabel("Execution Time (ms)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(fontsize=10)

        # -------------------------------------------------------------
        # Subplot 2: Empirical Execution Time (Log-Log Scale)
        # -------------------------------------------------------------
        for code_name in empirical_times:
            ax2.plot(
                sizes,
                empirical_times[code_name],
                label=f"{code_name}",
                color=colors[code_name],
                marker=markers[code_name],
                linewidth=2.2,
                markersize=6,
            )
        ax2.set_title("Execution Time vs. Input Size (Log-Log Scale)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Input Size (n) [Log Scale]", fontsize=11)
        ax2.set_ylabel("Execution Time (ms) [Log Scale]", fontsize=11)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", linestyle="--", alpha=0.6)
        ax2.legend(fontsize=10)

        # -------------------------------------------------------------
        # Subplot 3: Exact Operation Count vs Input Size (Log-Log Scale)
        # -------------------------------------------------------------
        for code_name in empirical_counts:
            ax3.plot(
                sizes,
                empirical_counts[code_name],
                label=f"{code_name} Empirical",
                color=colors[code_name],
                marker=markers[code_name],
                linewidth=2.2,
                markersize=6,
            )
            # Theoretical dashed curve
            ax3.plot(
                sizes,
                theoretical_counts[code_name],
                linestyle=":",
                color=colors[code_name],
                linewidth=1.8,
                alpha=0.8,
            )
        ax3.set_title("Operation Count x vs. Input Size (Log-Log Scale)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Input Size (n) [Log Scale]", fontsize=11)
        ax3.set_ylabel("Number of Additions x [Log Scale]", fontsize=11)
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.grid(True, which="both", linestyle="--", alpha=0.6)
        ax3.legend(fontsize=10)

        plt.suptitle(
            "DAA Lab 5 Q5: Empirical Execution Cost vs. Asymptotic Analysis",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save plot to images directory
        current_dir = Path(__file__).resolve().parent
        images_dir = current_dir.parent.parent / "images"
        save_path = (
            images_dir / "q5_tc_analysis.png"
            if images_dir.exists()
            else current_dir / "q5_tc_analysis.png"
        )
        plt.savefig(save_path, dpi=300)
        print(f"\n[Plot Saved] Figure saved to: {save_path}")

        # Sync with overleaf directory if present
        overleaf_images_dir = current_dir.parent.parent / "overleaf" / "lab_5" / "images"
        if overleaf_images_dir.exists():
            plt.savefig(overleaf_images_dir / "q5_tc_analysis.png", dpi=300)
            print(f"[Overleaf Sync] Figure synced to: {overleaf_images_dir / 'q5_tc_analysis.png'}")

        plt.show()
    except Exception as e:
        print(f"\n[Plotting Skipped or Display Unavailable]: {e}")


# =====================================================================
# 4. Interactive Driver
# =====================================================================

def interactive_run():
    """CLI Driver allowing interactive single-value tests or full static benchmark."""
    print("=" * 70)
    print("   DAA Lab 5 (Question 5): Execution Cost & Timing Program")
    print("=" * 70)
    print("Enter input size n (or press Enter / 0 for static benchmark suite):")
    user_input = input().strip()
    n = int(user_input) if user_input else 0

    if n == 0:
        run_static()
    else:
        print(f"\nEvaluating all 4 code snippets for n = {n} ...\n")

        # Code 1
        c1, t1 = measure_execution_time(code1, n, trials=5)
        th1 = theoretical_count_code1(n)
        print(f"(a) Code 1 [Theta(n^2)]:")
        print(f"    Measured x additions: {c1}")
        print(f"    Theoretical formula : n(n+1)/2 = {th1} (Match: {c1 == th1})")
        print(f"    Avg Execution Time  : {t1:.4f} ms\n")

        # Code 2
        c2, t2 = measure_execution_time(code2, n, trials=5)
        th2 = theoretical_count_code2(n)
        print(f"(b) Code 2 [Theta(n)] (using intended j = j // 2):")
        print(f"    Measured x additions: {c2}")
        print(f"    Theoretical formula : sum(floor(n/2^k)) = {th2} (Match: {c2 == th2})")
        print(f"    Avg Execution Time  : {t2:.4f} ms\n")

        # Code 3
        c3, t3 = measure_execution_time(code3, n, trials=3 if n >= 200 else 5)
        th3 = theoretical_count_code3(n)
        print(f"(c) Code 3 [Theta(n^3)]:")
        print(f"    Measured x additions: {c3}")
        print(f"    Theoretical formula : n(n+1)(2n+1)/6 = {th3} (Match: {c3 == th3})")
        print(f"    Avg Execution Time  : {t3:.4f} ms\n")

        # Code 4
        c4, t4 = measure_execution_time(code4, n, trials=5)
        th4 = theoretical_count_code4(n)
        print(f"(d) Code 4 [Theta(n log n)]:")
        print(f"    Measured x additions: {c4}")
        print(f"    Theoretical formula : (floor(log2 n) + 1) * n = {th4} (Match: {c4 == th4})")
        print(f"    Avg Execution Time  : {t4:.4f} ms\n")


if __name__ == "__main__":
    interactive_run()
