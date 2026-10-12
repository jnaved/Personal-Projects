# Karatsuba Large Integer Multiplication (Divide and Conquer)
import math
import random
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Increase integer string conversion limit for very large integers (Python 3.11+)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100000)


def conventional_multiply(x: int, y: int) -> int:
    """
    Conventional (Grade-School) Large Integer Multiplication.
    Performs digit-by-digit multiplication in base 10.
    
    Time Complexity:
        O(n * m) where n and m are the number of digits in x and y.
        When n = m, Time Complexity is O(n^2).
    """
    if x == 0 or y == 0:
        return 0

    sign = -1 if (x < 0) ^ (y < 0) else 1
    s1, s2 = str(abs(x)), str(abs(y))
    n, m = len(s1), len(s2)
    result = [0] * (n + m)

    # Multiply each digit of s1 with each digit of s2
    for i in range(n - 1, -1, -1):
        d1 = ord(s1[i]) - 48
        for j in range(m - 1, -1, -1):
            d2 = ord(s2[j]) - 48
            product = d1 * d2 + result[i + j + 1]
            result[i + j + 1] = product % 10
            result[i + j] += product // 10

    # Convert digit array back to integer
    res_val = 0
    for digit in result:
        res_val = res_val * 10 + digit

    return sign * res_val


def karatsuba(x: int, y: int) -> int:
    """
    Karatsuba Algorithm for Large Integer Multiplication (Divide and Conquer).
    
    Splits each integer of length n into two halves of length n/2:
        x = a * 10^(n/2) + b
        y = c * 10^(n/2) + d
    
    Instead of 4 subproblems (a*c, a*d, b*c, b*d), Karatsuba uses 3:
        z2 = a * c
        z0 = b * d
        z1 = (a + b) * (c + d) - z2 - z0 = a*d + b*c
    
    Combined Result:
        x * y = z2 * 10^(2 * (n/2)) + z1 * 10^(n/2) + z0
        
    Recurrence Relation:
        T(n) = 3 * T(n/2) + O(n)
        By Master Theorem (Case 1): T(n) = O(n^(log2(3))) ≈ O(n^1.585)
    """
    # Handle signs
    if x < 0 and y < 0:
        return karatsuba(-x, -y)
    elif x < 0:
        return -karatsuba(-x, y)
    elif y < 0:
        return -karatsuba(x, -y)

    # Base case: single digit multiplication
    if x < 10 or y < 10:
        return x * y

    # Determine maximum number of digits
    n = max(len(str(x)), len(str(y)))
    m = n // 2
    power_of_ten = 10 ** m

    # Divide: split numbers into high and low parts
    a, b = divmod(x, power_of_ten)
    c, d = divmod(y, power_of_ten)

    # Conquer: 3 recursive multiplications
    z0 = karatsuba(b, d)
    z2 = karatsuba(a, c)
    z1 = karatsuba(a + b, c + d) - z2 - z0

    # Combine: reconstruct product
    return z2 * (10 ** (2 * m)) + z1 * power_of_ten + z0


def generate_random_large_integer(num_digits: int, seed: int = None) -> int:
    """
    Generates a random integer with exactly num_digits digits.
    Ensures the leading digit is non-zero (1-9).
    """
    if seed is not None:
        random.seed(seed + num_digits)

    if num_digits <= 0:
        return 0
    if num_digits == 1:
        return random.randint(1, 9)

    first_digit = str(random.randint(1, 9))
    remaining_digits = "".join(str(random.randint(0, 9)) for _ in range(num_digits - 1))
    return int(first_digit + remaining_digits)


def run_static(sizes: list[int] = None):
    """
    Benchmarks Conventional Multiplication vs. Karatsuba Multiplication
    across increasing numbers of digits N (16, 32, 64, 128, 256, 512, 1024),
    measures execution times, validates correctness, and plots comparison graphs.
    """
    if sizes is None:
        sizes = [16, 32, 64, 128, 256, 512, 1024]

    conv_times = []
    karat_times = []
    results = []

    print("\n" + "=" * 96)
    print("      LARGE INTEGER MULTIPLICATION: CONVENTIONAL vs. KARATSUBA (DIVIDE & CONQUER)")
    print("=" * 96)
    print(f"{'Digits (N)':<12} | {'Conventional (ms)':<20} | {'Karatsuba (ms)':<16} | {'Speedup':<12} | {'Match'}")
    print("-" * 96)

    trials_map = {16: 15, 32: 10, 64: 10, 128: 5, 256: 3, 512: 2, 1024: 1}

    for sz in sizes:
        num_trials = trials_map.get(sz, 1 if sz >= 512 else 5)
        x = generate_random_large_integer(sz, seed=42)
        y = generate_random_large_integer(sz, seed=84)

        # 1. Conventional Multiplication Benchmark
        t0 = time.perf_counter()
        for _ in range(num_trials):
            conv_res = conventional_multiply(x, y)
        conv_time = ((time.perf_counter() - t0) / num_trials) * 1000

        # 2. Karatsuba Multiplication Benchmark
        t1 = time.perf_counter()
        for _ in range(num_trials):
            karat_res = karatsuba(x, y)
        karat_time = ((time.perf_counter() - t1) / num_trials) * 1000

        conv_times.append(conv_time)
        karat_times.append(karat_time)

        # Verification against Python native arbitrary-precision arithmetic
        is_match = (conv_res == karat_res == (x * y))
        speedup = (conv_time / karat_time) if karat_time > 0 else float("inf")

        results.append({
            "size": sz,
            "conv_time": conv_time,
            "karat_time": karat_time,
            "speedup": speedup,
            "match": is_match,
        })

        speedup_str = f"{speedup:.2f}x" if speedup != float("inf") else "N/A"
        match_str = "Yes" if is_match else "No"
        print(f"{sz:<12} | {conv_time:<20.4f} | {karat_time:<16.4f} | {speedup_str:<12} | {match_str}")

    print("=" * 96)

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Linear Scale (shows quadratic divergence of Conventional)
        ax1.plot(sizes, conv_times, label="Conventional $O(n^2)$", color="#ff7f0e", marker="s", linewidth=2.2, markersize=6)
        ax1.plot(sizes, karat_times, label=r"Karatsuba $O(n^{\log_2 3}) \approx O(n^{1.585})$", color="#1f77b4", marker="o", linewidth=2.2, markersize=6)
        ax1.set_title("Execution Time vs. Number of Digits (Linear Scale)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Digits (N)", fontsize=11)
        ax1.set_ylabel("Execution Time (ms)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(fontsize=10)

        # Subplot 2: Log-Log Scale (shows asymptotic slope comparison)
        ax2.plot(sizes, conv_times, label="Conventional $O(n^2)$", color="#ff7f0e", marker="s", linewidth=2.2, markersize=6)
        ax2.plot(sizes, karat_times, label=r"Karatsuba $O(n^{\log_2 3}) \approx O(n^{1.585})$", color="#1f77b4", marker="o", linewidth=2.2, markersize=6)
        ax2.set_title("Execution Time vs. Number of Digits (Log-Log Scale)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Digits (N) [Log Scale]", fontsize=11)
        ax2.set_ylabel("Execution Time (ms) [Log Scale]", fontsize=11)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", linestyle="--", alpha=0.6)
        ax2.legend(fontsize=10)

        plt.suptitle("Large Integer Multiplication: Conventional vs. Karatsuba Algorithm", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save plot to images directory
        current_dir = Path(__file__).resolve().parent
        images_dir = current_dir.parent.parent / "images"
        save_path = images_dir / "karatsuba_tc_analysis.png" if images_dir.exists() else current_dir / "karatsuba_tc_analysis.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n[Plot Saved] Figure saved to: {save_path}")

        # Also sync with overleaf directory if present
        overleaf_images_dir = current_dir.parent.parent / "overleaf" / "lab_6" / "images"
        if overleaf_images_dir.exists():
            plt.savefig(overleaf_images_dir / "karatsuba_tc_analysis.png", dpi=300)
            print(f"[Overleaf Sync] Figure synced to: {overleaf_images_dir / 'karatsuba_tc_analysis.png'}")

        plt.show()
    except Exception as e:
        print(f"\n[Plotting Skipped or Display Unavailable]: {e}")


def karatsuba_multiplier():
    """Interactive CLI driver matching closest_pair.py pattern."""
    print("Enter the number of digits to execute: default(0 for static benchmark)")
    n_input = input().strip()
    n = int(n_input) if n_input else 0

    if not n:
        run_static()
    else:
        print(f"\nGenerating two random {n}-digit integers...")
        x = generate_random_large_integer(n)
        y = generate_random_large_integer(n)

        # Truncate preview if integer is very long
        x_str = str(x)
        y_str = str(y)
        preview_x = x_str if len(x_str) <= 40 else f"{x_str[:20]}...{x_str[-20:]} ({len(x_str)} digits)"
        preview_y = y_str if len(y_str) <= 40 else f"{y_str[:20]}...{y_str[-20:]} ({len(y_str)} digits)"

        print(f"X = {preview_x}")
        print(f"Y = {preview_y}\n")

        # Conventional
        t0 = time.perf_counter()
        conv_res = conventional_multiply(x, y)
        t_conv = (time.perf_counter() - t0) * 1000

        # Karatsuba
        t1 = time.perf_counter()
        karat_res = karatsuba(x, y)
        t_karat = (time.perf_counter() - t1) * 1000

        # Native Python multiplication
        native_res = x * y

        is_match = (conv_res == karat_res == native_res)
        res_str = str(karat_res)
        preview_res = res_str if len(res_str) <= 50 else f"{res_str[:25]}...{res_str[-25:]} ({len(res_str)} digits)"

        print(f"Product             : {preview_res}")
        print(f"Verification Match  : {is_match}")
        print(f"Conventional Time   : {t_conv:.4f} ms")
        print(f"Karatsuba Time      : {t_karat:.4f} ms")
        speedup = (t_conv / t_karat) if t_karat > 0 else float("inf")
        print(f"Karatsuba Speedup   : {speedup:.2f}x")


if __name__ == "__main__":
    karatsuba_multiplier()
