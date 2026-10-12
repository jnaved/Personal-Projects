"""
Assignment 1:
Design a divide-and-conquer algorithm for computing the number of levels in a binary tree.
(In particular, the algorithm must return 0 and 1 for the empty and single-node trees, respectively.)
Analyze the time efficiency class of your algorithm.
Implement the algorithm and find out the time taken for empty, one-level, two-level and three-level binary trees.
Compare it with the theoretical analysis.
"""

from __future__ import annotations
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


class TreeNode:
    """Represents a node in a binary tree."""
    def __init__(self, val: int = 0, left: TreeNode | None = None, right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right


def compute_levels(root: TreeNode | None) -> int:
    """
    Divide-and-Conquer algorithm for computing the number of levels in a binary tree.

    Base Cases:
      - Empty tree: 0 levels
      - Single-node tree: 1 level

    Divide:
      - Recursively compute the number of levels of the left subtree
      - Recursively compute the number of levels of the right subtree

    Combine:
      - Total levels = 1 + max(left_levels, right_levels)

    Time Complexity: Theta(N) where N is the number of nodes in the tree.
    Space Complexity: O(H) where H is the height of the tree (recursion call stack).
    """
    if root is None:
        return 0

    if root.left is None and root.right is None:
        return 1

    left_levels = compute_levels(root.left)
    right_levels = compute_levels(root.right)

    return 1 + max(left_levels, right_levels)


def build_sample_trees() -> dict[str, tuple[TreeNode | None, int]]:
    """Builds sample binary trees for 0, 1, 2, and 3 levels."""
    # 0 levels: Empty tree
    tree_0 = None

    # 1 level: Single-node tree
    tree_1 = TreeNode(1)

    # 2 levels: Root with children
    tree_2 = TreeNode(1, left=TreeNode(2), right=TreeNode(3))

    # 3 levels: Full binary tree of height 3 (7 nodes)
    tree_3 = TreeNode(
        1,
        left=TreeNode(2, left=TreeNode(4), right=TreeNode(5)),
        right=TreeNode(3, left=TreeNode(6), right=TreeNode(7)),
    )

    return {
        "Empty (0-level)": (tree_0, 0),
        "One-level": (tree_1, 1),
        "Two-level": (tree_2, 2),
        "Three-level": (tree_3, 3),
    }


def benchmark_tree_levels(iterations: int = 200_000, show_plot: bool = True):
    """
    Measures empirical execution time for empty, 1-level, 2-level, and 3-level binary trees.
    """
    trees = build_sample_trees()
    results = {}

    print("=" * 65)
    print(f"{'Tree Type':<20} | {'Expected':<10} | {'Computed':<10} | {'Avg Time (ns)':<15}")
    print("=" * 65)

    names = []
    times_ns = []
    level_counts = []

    for name, (tree, expected_levels) in trees.items():
        computed = compute_levels(tree)
        assert computed == expected_levels, f"Mismatch for {name}: expected {expected_levels}, got {computed}"

        # Warm up
        for _ in range(10_000):
            compute_levels(tree)

        # Benchmark
        start = time.perf_counter_ns()
        for _ in range(iterations):
            compute_levels(tree)
        total_time_ns = time.perf_counter_ns() - start
        avg_time_ns = total_time_ns / iterations

        results[name] = avg_time_ns
        names.append(name)
        times_ns.append(avg_time_ns)
        level_counts.append(expected_levels)

        print(f"{name:<20} | {expected_levels:<10} | {computed:<10} | {avg_time_ns:<15.2f}")

    print("=" * 65)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#9467bd", "#1f77b4", "#2ca02c", "#d62728"]
    bars = ax1.bar(names, times_ns, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_title("Average Execution Time per Tree Structure", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Execution Time (nanoseconds)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f} ns",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.plot(level_counts, times_ns, marker="o", color="#d62728", linewidth=2.2, markersize=8)
    ax2.set_title("Number of Levels vs. Execution Time", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Levels (Height)", fontsize=11)
    ax2.set_ylabel("Execution Time (nanoseconds)", fontsize=11)
    ax2.set_xticks(level_counts)
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle("Binary Tree Level Computation (Divide and Conquer Analysis)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    for d in IMAGES_DIRS:
        save_path = d / "tree_levels_timing.png"
        plt.savefig(save_path, dpi=300)
        print(f"[Plot Saved] {save_path}")

    if show_plot:
        plt.show()

    return results


if __name__ == "__main__":
    benchmark_tree_levels(show_plot=False)
