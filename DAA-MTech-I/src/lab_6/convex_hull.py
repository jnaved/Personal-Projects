"""
Assignment 4:
The convex hull of a set of points in a Euclidean space is the smallest convex polygon that
encloses all of the points. In two dimensions (2D), the convex hull is a convex polygon.
Given an array of coordinates x, y on a 2-D plane, print the set of coordinates that form
the convex hull.

Examples:
Input: points[] = {(0, 0), (0, 4), (-4, 0), (5, 0), (0, -6), (1, 0)}
Output: (-4, 0), (5, 0), (0, -6), (0, 4)

Input: points list = {{1, 2}, {3, 1}, {5, 6}}
Output: {{1, 2}, {3, 1}, {5, 6}}
"""

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

Point = tuple[float, float]


def cross_product(o: Point, a: Point, b: Point) -> float:
    """
    Computes 2D cross product of vectors OA and OB.
    Returns:
      > 0: Counterclockwise turn (left turn)
      < 0: Clockwise turn (right turn)
      = 0: Collinear points
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def compute_convex_hull(points: list[Point]) -> list[Point]:
    """
    Computes the convex hull of a set of 2D points using Andrew's Monotone Chain algorithm.
    Time Complexity: O(n log n) due to initial sorting.
    Space Complexity: O(n) to store upper and lower hulls.
    """
    unique_points = sorted(set(points))
    n = len(unique_points)

    if n <= 2:
        return unique_points

    # 1. Build Lower Hull
    lower: list[Point] = []
    for p in unique_points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # 2. Build Upper Hull
    upper: list[Point] = []
    for p in reversed(unique_points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper hulls (omitting duplicate endpoints)
    return lower[:-1] + upper[:-1]


def plot_convex_hull(points: list[Point], hull: list[Point], title: str, save_filename: str, show_plot: bool = True):
    """Visualizes the 2D point cloud and its surrounding convex hull polygon."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all input points
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    ax.scatter(px, py, color="#1f77b4", s=90, zorder=3, label="Input Points")

    # Annotate points
    for p in points:
        ax.annotate(f"({p[0]}, {p[1]})", (p[0], p[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, alpha=0.85)

    # Plot convex hull boundary
    hull_closed = hull + [hull[0]]
    hx = [p[0] for p in hull_closed]
    hy = [p[1] for p in hull_closed]
    ax.plot(hx, hy, color="#2ca02c", linewidth=2.5, linestyle="-", label="Convex Hull Boundary")
    ax.fill(hx, hy, color="#2ca02c", alpha=0.15)

    # Highlight hull vertices
    ax.scatter([p[0] for p in hull], [p[1] for p in hull], color="#d62728", s=130, zorder=4, label="Hull Vertices")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X coordinate", fontsize=11)
    ax.set_ylabel("Y coordinate", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()

    for d in IMAGES_DIRS:
        save_path = d / save_filename
        plt.savefig(save_path, dpi=300)
        print(f"[Plot Saved] {save_path}")

    if show_plot:
        plt.show()


def run_assignment_examples(show_plot: bool = True):
    """Executes the specific examples provided in Assignment 4."""
    print("=" * 65)
    print("ASSIGNMENT 4: 2D CONVEX HULL EXECUTION")
    print("=" * 65)

    # Example 1
    ex1_points = [(0, 0), (0, 4), (-4, 0), (5, 0), (0, -6), (1, 0)]
    ex1_hull = compute_convex_hull(ex1_points)
    print(f"Example 1 Input:  {ex1_points}")
    print(f"Example 1 Output: {ex1_hull}")
    expected_set1 = {(-4, 0), (5, 0), (0, -6), (0, 4)}
    assert set(ex1_hull) == expected_set1, f"Expected {expected_set1}, got {set(ex1_hull)}"
    print(f"Match Expected Set: {set(ex1_hull) == expected_set1} -> {expected_set1}\n")

    # Example 2
    ex2_points = [(1, 2), (3, 1), (5, 6)]
    ex2_hull = compute_convex_hull(ex2_points)
    print(f"Example 2 Input:  {ex2_points}")
    print(f"Example 2 Output: {ex2_hull}")
    expected_set2 = {(1, 2), (3, 1), (5, 6)}
    assert set(ex2_hull) == expected_set2, f"Expected {expected_set2}, got {set(ex2_hull)}"
    print(f"Match Expected Set: {set(ex2_hull) == expected_set2} -> {expected_set2}\n")

    # Visualize Example 1
    plot_convex_hull(
        ex1_points,
        ex1_hull,
        title="2D Convex Hull: Example 1 Points & Enclosing Polygon",
        save_filename="convex_hull_visualization.png",
        show_plot=show_plot,
    )
    print("=" * 65)


if __name__ == "__main__":
    run_assignment_examples(show_plot=False)
