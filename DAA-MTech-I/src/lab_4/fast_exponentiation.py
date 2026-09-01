'''
Computing a power n for large n
'''
import matplotlib.pyplot as plt
import time
from pathlib import Path


def rep_multiplication(a: int, n: int):
    res = 1
    for i in range(1, n+1):
        res *= a
    return res


def recursive_exponentiation(a: int, n: int):
    if n == 0:
        return 1
    elif n&1:
        return a*recursive_exponentiation(a=a, n=n-1)
    else:
        return recursive_exponentiation(a=a, n=n//2)*recursive_exponentiation(a=a, n=n//2)


def dac_exponentiation(a: int, n: int):
    if n==0:
        return 1
    elif n==1:
        return a
    
    val = dac_exponentiation(a=a, n=n//2)
    if n&1:
        return val*val*a
    else:
        return val*val


def run_and_plot():
    powers = [10, 50, 100, 250, 500, 1000, 10000, 100000]
    a = 5

    dac_times = []
    iter_times = []
    rec_times = []

    for n in powers:
        # 1. Divide and Conquer Approach O(log n)
        t0 = time.perf_counter()
        dac_exponentiation(a=a, n=n)
        dac_times.append((time.perf_counter() - t0) * 1000)

        # 2. Repeated Multiplication Approach O(n)
        t1 = time.perf_counter()
        rep_multiplication(a=a, n=n)
        iter_times.append((time.perf_counter() - t1) * 1000)

        # 3. Recursive Exponentiation Approach
        t2 = time.perf_counter()
        recursive_exponentiation(a=a, n=n)
        rec_times.append((time.perf_counter() - t2) * 1000)

    # --- Plotting (Single Comparative Graph) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(powers, dac_times, label="Divide and Conquer (DAC) O(log N)", color="#1f77b4", marker="o", linewidth=2.5, markersize=7)
    ax.plot(powers, iter_times, label="Repeated Multiplication O(N)", color="#ff7f0e", marker="s", linewidth=2, markersize=6)
    ax.plot(powers, rec_times, label="Recursive Exponentiation O(N)", color="#2ca02c", marker="^", linewidth=2, markersize=6)

    ax.set_title("Fast Exponentiation: Time Complexity Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Exponent / Power (N)", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    current_dir = Path(__file__).resolve().parent
    images_dir = current_dir.parent.parent / "images"
    save_path = images_dir / "fast_exponentiation.png" if images_dir.exists() else current_dir / "fast_exponentiation.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Plot Saved] Figure saved to: {save_path}")

    # Also sync with overleaf if present
    overleaf_images_dir = current_dir.parent.parent / "overleaf" / "lab_4" / "images"
    if overleaf_images_dir.exists():
        plt.savefig(overleaf_images_dir / "fast_exponentiation.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    print("Enter the value of a and n (space-separated) [e.g. 5 20]:")
    user_input = input().strip()
    if user_input:
        a, n = list(map(int, user_input.split()))
    else:
        a, n = 5, 20

    t1 = time.perf_counter()
    dac_exponentiation(a=a, n=n)
    dac_t = (time.perf_counter() - t1) * 1000
    print(f"DAC Approach Time: {dac_t:.6f} ms")

    t2 = time.perf_counter()
    rep_multiplication(a=a, n=n)
    rep_t = (time.perf_counter() - t2) * 1000
    print(f"Repeated Multiplication Time: {rep_t:.6f} ms")

    if n <= 10000:
        t3 = time.perf_counter()
        recursive_exponentiation(a=a, n=n)
        rec_t = (time.perf_counter() - t3) * 1000
        print(f"Recursive Exponentiation Time: {rec_t:.6f} ms")
    else:
        print("Recursive Exponentiation Skipped (N > 10000 exceeds recursion limit)")

    run_and_plot()