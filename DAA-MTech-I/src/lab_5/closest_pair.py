# Find closest pair of points in 2-D plane
import math
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt

def find_dist(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.sqrt(
        (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    )


def brute_force(points: list[tuple[int, int]]):
    n = len(points)
    min_dist: float = float('inf')
    min_pair = []
    for i in range(n-1):
        p1 = points[i]
        for j in range(i+1, n):
            p2 = points[j]
            dist = find_dist(p1=p1, p2=p2)

            if dist < min_dist:
                min_dist = dist
                min_pair = [p1, p2]
    
    return min_dist, min_pair


def strip_closest(strip: tuple[int, int], d: float):
    min_dist = d
    min_pair = []
    strip.sort(key=lambda x: x[1])

    for i in range(len(strip)):
        for j in range(i+1, len(strip)):
            if strip[j][1] - strip[i][1] >= min_dist:
                break

            dist = find_dist(p1=strip[i], p2=strip[j])
            if dist < min_dist:
                min_dist = dist
                min_pair = [strip[i], strip[j]]
    
    return min_dist, min_pair

def dac(points: list[tuple[int, int]]):
    points.sort()
    n = len(points)

    if n<=3:
        return brute_force(points=points)

    mid = n//2
    mid_point = points[mid]

    d_left, left_sub = dac(points=points[:mid])
    d_right, right_sub = dac(points=points[mid:])

    d, min_pair = [d_left, left_sub] if d_left < d_right else [d_right, right_sub]

    strip = [p for p in points if abs(p[0] - mid_point[0]) < d]
    d_strip, strip_pair = strip_closest(strip=strip, d=d)
    return (d_strip, strip_pair) if strip_pair else (d, min_pair)


def generate_random_points(count: int, seed: int = 42) -> list[tuple[int, int]]:
    """Generates a list of distinct 2D points with coordinates in [0, 100000]."""
    random.seed(seed + count)
    points = set()
    while len(points) < count:
        points.add((random.randint(0, 100000), random.randint(0, 100000)))
    return list(points)


def run_static(sizes: list[int] = None):
    """
    Generates points for different input sizes (10, 100, 1000, etc.),
    compares the empirical time complexity of Brute Force vs. Divide & Conquer (DAC),
    and plots the benchmark results.
    """
    if sizes is None:
        sizes = [10, 50, 100, 200, 500, 1000]

    bf_times = []
    dac_times = []
    results = []

    print("\n" + "=" * 85)
    print("      CLOSEST PAIR OF POINTS: BRUTE FORCE vs. DIVIDE AND CONQUER (DAC)")
    print("=" * 85)
    print(f"{'Size (N)':<10} | {'Brute Force (ms)':<18} | {'DAC (ms)':<14} | {'Speedup':<12} | {'Min Distance':<14} | {'Match'}")
    print("-" * 85)

    for sz in sizes:
        points = generate_random_points(count=sz)

        # 1. Brute Force Benchmark
        t0 = time.perf_counter()
        bf_dist, bf_pair = brute_force(points=points.copy())
        bf_time = (time.perf_counter() - t0) * 1000

        # 2. Divide and Conquer Benchmark
        t1 = time.perf_counter()
        dac_dist, dac_pair = dac(points=points.copy())
        dac_time = (time.perf_counter() - t1) * 1000

        bf_times.append(bf_time)
        dac_times.append(dac_time)

        is_match = math.isclose(bf_dist, dac_dist, rel_tol=1e-5)
        speedup = (bf_time / dac_time) if dac_time > 0 else float('inf')

        results.append({
            "size": sz,
            "bf_time": bf_time,
            "dac_time": dac_time,
            "speedup": speedup,
            "min_dist": dac_dist,
            "match": is_match
        })

        speedup_str = f"{speedup:.2f}x" if speedup != float('inf') else "N/A"
        match_str = "Yes" if is_match else "No"
        print(f"{sz:<10} | {bf_time:<18.4f} | {dac_time:<14.4f} | {speedup_str:<12} | {dac_dist:<14.4f} | {match_str}")

    print("=" * 85)

    # --- Plotting Benchmark Comparison ---
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Linear Scale (shows quadratic divergence of Brute Force)
        ax1.plot(sizes, bf_times, label="Brute Force $O(n^2)$", color="#ff7f0e", marker="s", linewidth=2.2, markersize=6)
        ax1.plot(sizes, dac_times, label="Divide & Conquer $O(n \\log^2 n)$", color="#1f77b4", marker="o", linewidth=2.2, markersize=6)
        ax1.set_title("Execution Time vs. Input Size (Linear Scale)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Points (N)", fontsize=11)
        ax1.set_ylabel("Execution Time (ms)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(fontsize=10)

        # Subplot 2: Log-Log Scale (shows asymptotic growth slopes)
        ax2.plot(sizes, bf_times, label="Brute Force $O(n^2)$", color="#ff7f0e", marker="s", linewidth=2.2, markersize=6)
        ax2.plot(sizes, dac_times, label="Divide & Conquer $O(n \\log^2 n)$", color="#1f77b4", marker="o", linewidth=2.2, markersize=6)
        ax2.set_title("Execution Time vs. Input Size (Log-Log Scale)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Points (N) [Log Scale]", fontsize=11)
        ax2.set_ylabel("Execution Time (ms) [Log Scale]", fontsize=11)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", linestyle="--", alpha=0.6)
        ax2.legend(fontsize=10)

        plt.suptitle("Closest Pair of Points: Time Complexity Analysis (Brute Force vs. DAC)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save plot to images directory
        current_dir = Path(__file__).resolve().parent
        images_dir = current_dir.parent.parent / "images"
        save_path = images_dir / "closest_pair_tc_analysis.png" if images_dir.exists() else current_dir / "closest_pair_tc_analysis.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n[Plot Saved] Figure saved to: {save_path}")

        # Also sync with overleaf if present
        overleaf_images_dir = current_dir.parent.parent / "overleaf" / "lab_5" / "images"
        if overleaf_images_dir.exists():
            plt.savefig(overleaf_images_dir / "closest_pair_tc_analysis.png", dpi=300)

        plt.show()
    except Exception as e:
        print(f"\n[Plotting Skipped or Display Unavailable]: {e}")


def closest_pairs():
    print("Enter the number of points to execute: default(0)")
    n = int(input().strip() or 0)
    if not n:
        run_static()
    else:
        points = []
        for idx in range(n):
            x, y = list(map(int, input(f"Enter the element {idx+1} :").split()))
            points.append((x, y))

        min_dist, min_pair = dac(points=points)
        print(f"Minimum distance found between points is {min_dist} and the pair is {min_pair}")

closest_pairs()