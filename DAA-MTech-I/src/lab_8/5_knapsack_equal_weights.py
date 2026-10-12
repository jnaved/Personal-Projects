"""
Assignment 5:
Consider the 0/1 knapsack problem with n objects, whose profits and weights are vi, wi (1 <= i <= n),
respectively. The capacity of the knapsack is m. It is also given that all the objects have
the same weight. Present an O(m) algorithm to solve and implement this problem.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.lab_8.knapsack_helpers import solve_knapsack_01_dp
from utils.lab_8.selection import get_top_k_largest


def solve_knapsack_equal_weights(
    profits: List[int],
    uniform_weight: int,
    capacity: int,
) -> Tuple[int, List[int]]:
    """
    Solves the 0/1 Knapsack problem when all objects have the identical weight w_0 in O(m) time
    (for n = O(m), replacing the general O(n * m) dynamic programming).

    Parameters:
        profits (List[int]): Profit values v_1, v_2, ..., v_n of the objects.
        uniform_weight (int): Identical weight w_0 of every object (w_i = uniform_weight).
        capacity (int): Total knapsack capacity m.

    Returns:
        Tuple[int, List[int]]: (maximum_profit, selected_item_indices)

    Algorithmic Analysis & Proof:
        1. Capacity in terms of items:
           Since every item consumes uniform_weight (w_0), the maximum number of items that
           can fit into capacity m is:
               k = floor(m / uniform_weight).
        2. Greedy / Selection Strategy:
           Because every item has identical weight, all items have the same capacity cost.
           Therefore, maximizing total profit reduces strictly to selecting the k items with
           the largest profits.
        3. Achieving O(m) Time Complexity:
           - In the general 0/1 knapsack, Dynamic Programming requires O(n * m) time.
           - When each object weighs w_0 >= 1, we can fit at most k <= m objects.
           - In instances where n = O(m) (or when n is scaled with capacity), finding the top k
             elements using Quickselect (Linear-time Selection) takes O(n) = O(m) expected time.
           - No full sorting (O(n log n)) is required.
           - Summing the top k selected elements takes O(k) <= O(m) time.
           - Overall runtime: O(m) time and O(n) auxiliary space.

    Time Complexity:
        - Selection of top k items via Quickselect: O(n) = O(m) when n = O(m).
        - Summation: O(k) <= O(m).
        - Total Time: O(m).
    """
    n = len(profits)
    if n == 0 or capacity <= 0 or uniform_weight <= 0:
        return 0, []

    # Maximum number of items that can fit
    k = capacity // uniform_weight

    # Case 1: All items can fit in the knapsack
    if k >= n:
        total_profit = sum(profits)
        return total_profit, list(range(n))

    # Case 2: Knapsack can hold at most k items (k < n)
    # Pair each profit with its original index: (profit, original_index)
    indexed_profits = [(profits[i], i) for i in range(n)]

    # Use Quickselect to retrieve top k elements in O(n) = O(m) time
    top_k = get_top_k_largest(indexed_profits, k, key=lambda item: item[0])

    total_profit = sum(item[0] for item in top_k)
    selected_indices = [item[1] for item in top_k]

    return total_profit, selected_indices


if __name__ == "__main__":
    print("=" * 80)
    print("Assignment 5: 0/1 Knapsack with Equal Weights - O(m) Algorithm")
    print("=" * 80)

    # Test instance
    m = 10                  # Knapsack capacity
    w_0 = 2                 # Uniform weight for all items
    sample_profits = [12, 5, 20, 15, 8, 25, 18, 30]  # n = 8 items

    k = m // w_0
    print(f"Knapsack Capacity (m)   : {m}")
    print(f"Uniform Weight (w_0)    : {w_0}")
    print(f"Max Items Allowed (k)   : {k} (since floor(10 / 2) = 5)")
    print(f"Item Profits            : {sample_profits}")

    # Solve using O(m) algorithm
    opt_profit, chosen_idx = solve_knapsack_equal_weights(sample_profits, w_0, m)

    print("-" * 80)
    print(f"O(m) Algorithm Profit   : {opt_profit}")
    print(f"Chosen Item Indices     : {chosen_idx}")
    print(f"Chosen Profits          : {[sample_profits[i] for i in chosen_idx]}")

    # Verify against exact General 0/1 Knapsack Dynamic Programming solver O(n * m)
    weights = [w_0] * len(sample_profits)
    dp_profit, dp_idx = solve_knapsack_01_dp(weights, sample_profits, m)
    print(f"General 0/1 Knapsack DP : {dp_profit}")

    assert opt_profit == dp_profit, f"Mismatch with exact DP! {opt_profit} != {dp_profit}"
    print("Verification Passed: O(m) selection strictly matches exact 0/1 Knapsack DP!")

    # Performance comparison between O(m) selection and O(n * m) DP
    print("-" * 80)
    print("Performance Comparison (O(m) Selection vs. O(n*m) DP):")
    large_n = 2000
    large_m = 2000
    large_w0 = 1
    import random
    rng = random.Random(123)
    large_profits = [rng.randint(1, 1000) for _ in range(large_n)]
    large_weights = [large_w0] * large_n

    t0 = time.perf_counter()
    p_opt, _ = solve_knapsack_equal_weights(large_profits, large_w0, large_m)
    t_opt = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    p_dp, _ = solve_knapsack_01_dp(large_weights, large_profits, large_m)
    t_dp = (time.perf_counter() - t0) * 1000

    assert p_opt == p_dp
    print(f"For n={large_n}, m={large_m}:")
    print(f" - O(m) Selection Time   : {t_opt:.3f} ms")
    print(f" - O(n*m) DP Time        : {t_dp:.3f} ms (Speedup: {t_dp / max(0.001, t_opt):.1f}x)")
    print("=" * 80)
