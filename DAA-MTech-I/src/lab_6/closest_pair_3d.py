"""
Assignment 5:
Solve the shortest distance between two points in 3D.
[hint: You can use a similar approach with a few modifications that you have used in the 2D version of this problem]
"""

import math
import random
import time
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

Point3D = tuple[float, float, float]


def dist_3d(p1: Point3D, p2: Point3D) -> float:
    """Computes Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def brute_force_3d(points: list[Point3D]) -> tuple[float, tuple[Point3D, Point3D]]:
    """
    Brute-force approach to find the closest pair in 3D.
    Time Complexity: Theta(n^2)
    """
    n = len(points)
    min_d = float("inf")
    best_pair = (points[0], points[1])

    for i in range(n - 1):
        for j in range(i + 1, n):
            d = dist_3d(points[i], points[j])
            if d < min_d:
                min_d = d
                best_pair = (points[i], points[j])

    return min_d, best_pair


def closest_pair_3d_rec(px: list[Point3D]) -> tuple[float, tuple[Point3D, Point3D]]:
    """
    Divide-and-Conquer recursive solver for 3D Closest Pair.
    Extends the 2D planar divide-and-conquer to 3-dimensional Euclidean space.

    Time Complexity: O(n log^2 n)
    Space Complexity: O(n)
    """
    n = len(px)
    if n <= 3:
        return brute_force_3d(px)

    mid = n // 2
    mid_point = px[mid]

    left_px = px[:mid]
    right_px = px[mid:]

    dl, pair_l = closest_pair_3d_rec(left_px)
    dr, pair_r = closest_pair_3d_rec(right_px)

    if dl < dr:
        d = dl
        best_pair = pair_l
    else:
        d = dr
        best_pair = pair_r

    # 3D Slab: Filter points whose x-coordinate is within d of the dividing plane x = mid_point[0]
    slab = [p for p in px if abs(p[0] - mid_point[0]) < d]

    # Sort slab by y-coordinate
    slab.sort(key=lambda p: p[1])
    slab_len = len(slab)

    # In 3D space, by sphere packing within a box of dimensions 2d x d x 2d,
    # at most a constant number of points can lie within distance d.
    for i in range(slab_len):
        pi = slab[i]
        for j in range(i + 1, slab_len):
            pj = slab[j]
            # Prune if y-difference exceeds current minimum
            if (pj[1] - pi[1]) >= d:
                break
            # Prune if z-difference exceeds current minimum
            if abs(pj[2] - pi[2]) >= d:
                continue

            dist = dist_3d(pi, pj)
            if dist < d:
                d = dist
                best_pair = (pi, pj)

    return d, best_pair


def closest_pair_3d(points: list[Point3D]) -> tuple[float, tuple[Point3D, Point3D]]:
    """Entry point for 3D Closest Pair Divide and Conquer."""
    px = sorted(points, key=lambda p: p[0])
    return closest_pair_3d_rec(px)


def visualize_closest_pair_3d(points: list[Point3D], closest_pair: tuple[Point3D, Point3D], distance: float, show_plot: bool = True):
    """Renders a 3D scatter plot highlighting the closest pair of points."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    # Plot point cloud
    ax.scatter(xs, ys, zs, c="#1f77b4", s=30, alpha=0.6, label="3D Point Cloud")

    p1, p2 = closest_pair
    # Highlight closest pair
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="#d62728", s=140, edgecolors="black", linewidths=1.5, zorder=5, label="Closest Pair")

    # Draw connecting line between closest pair
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="#d62728", linewidth=3.0, linestyle="--", label=f"Shortest Distance: {distance:.4f}")

    ax.set_title("Shortest Distance Between Two Points in 3D Space", fontsize=13, fontweight="bold")
    ax.set_xlabel("X Axis", fontsize=10)
    ax.set_ylabel("Y Axis", fontsize=10)
    ax.set_zlabel("Z Axis", fontsize=10)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    for d in IMAGES_DIRS:
        save_path = d / "closest_pair_3d_visualization.png"
        plt.savefig(save_path, dpi=300)
        print(f"[Plot Saved] {save_path}")

    if show_plot:
        plt.show()


def benchmark_3d_closest_pair(show_plot: bool = True):
    """Tests 3D closest pair on increasing point set sizes and compares with brute force."""
    rng = random.Random(42)
    test_sizes = [50, 100, 200, 500, 1000, 2000, 5000]

    print("=" * 75)
    print(f"{'Points (N)':<12} | {'Min Distance':<15} | {'Brute Force (ms)':<18} | {'Divide & Conquer (ms)':<22}")
    print("=" * 75)

    for n in test_sizes:
        points = [(rng.uniform(0, 1000), rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]

        # Divide & Conquer
        t0 = time.perf_counter()
        d_dc, pair_dc = closest_pair_3d(points)
        t_dc = (time.perf_counter() - t0) * 1000

        # Brute Force
        if n <= 1000:
            t0 = time.perf_counter()
            d_bf, pair_bf = brute_force_3d(points)
            t_bf = (time.perf_counter() - t0) * 1000
            assert math.isclose(d_dc, d_bf, rel_tol=1e-9), f"Discrepancy: DC={d_dc}, BF={d_bf}"
            bf_str = f"{t_bf:.3f}"
        else:
            bf_str = "N/A (O(n²))"

        print(f"{n:<12} | {d_dc:<15.4f} | {bf_str:<18} | {t_dc:<22.3f}")

    print("=" * 75)

    # Visualize for a 100-point sample
    sample_points = [(rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(100)]
    min_dist, pair = closest_pair_3d(sample_points)
    print(f"\nClosest pair found in 3D: {pair[0]} <-> {pair[1]} with distance = {min_dist:.4f}")
    visualize_closest_pair_3d(sample_points, pair, min_dist, show_plot=show_plot)


if __name__ == "__main__":
    benchmark_3d_closest_pair(show_plot=False)
