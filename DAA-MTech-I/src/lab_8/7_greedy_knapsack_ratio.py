"""
Assignment 7:
Let F(I) be the solution’s value generated on a knapsack problem instance I by Greedy-
Knapsack when the objects are processed in non-increasing order of their profit values. Let
F*(I) be the value of the optimal solution for this instance. How large can the ratio F*(I)/F(I)
get? Show with a parameterized implementation of this problem where, apart from other data,
the ratio F*(I)/F(I) is given as input in terms of a positive integer.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.lab_8.knapsack_helpers import solve_knapsack_01_dp, solve_knapsack_greedy_by_profit


def generate_parameterized_knapsack_instance(target_ratio: int) -> Tuple[List[int], List[int], int]:
    """
    Constructs a parameterized knapsack instance I such that the ratio F*(I) / F(I)
    equals exactly target_ratio (a positive integer R >= 1).

    Construction:
        Let R = target_ratio.
        Knapsack Capacity: M = 2 * R
        Number of items: 2 * R + 1
        - Item 0 (Heavy): Profit p_0 = 2, Weight w_0 = 2 * R
        - Items 1 to 2*R (Light): Each item has Profit p_i = 1, Weight w_i = 1

    Behavior:
        1. When processed in non-increasing order of profit values:
           p_0 = 2 > p_1 = p_2 = ... = p_{2R} = 1.
           Greedy-Knapsack processes Item 0 first.
           Item 0 fits (weight 2R <= 2R), using up the entire capacity M.
           Remaining capacity becomes 0. None of the other items can be picked.
           Greedy profit: F(I) = 2.
        2. Optimal Solution F*(I):
           Does not pick Item 0. Instead, picks all 2*R light items.
           Total weight = 2*R * 1 = 2*R <= M.
           Optimal profit: F*(I) = 2 * R * 1 = 2 * R.
        3. Ratio:
           F*(I) / F(I) = (2 * R) / 2 = R = target_ratio.
    """
    if target_ratio < 1:
        raise ValueError("target_ratio must be a positive integer (>= 1).")

    capacity = 2 * target_ratio

    # Item 0: profit = 2, weight = 2 * R
    weights = [2 * target_ratio]
    profits = [2]

    # 2*R light items: profit = 1, weight = 1
    for _ in range(2 * target_ratio):
        weights.append(1)
        profits.append(1)

    return weights, profits, capacity


def analyze_greedy_knapsack_ratio(target_ratio: int, verbose: bool = True) -> Dict[str, any]:
    """
    Demonstrates the unbounded ratio F*(I) / F(I) on a parameterized instance.
    """
    weights, profits, capacity = generate_parameterized_knapsack_instance(target_ratio)

    # 1. Greedy-Knapsack by non-increasing profit order
    f_greedy, greedy_chosen = solve_knapsack_greedy_by_profit(weights, profits, capacity)

    # 2. Optimal Solution via Exact Dynamic Programming
    f_optimal, opt_chosen = solve_knapsack_01_dp(weights, profits, capacity)

    actual_ratio = f_optimal / f_greedy

    if verbose:
        print(f"\nTarget Ratio R         : {target_ratio}")
        print(f"Knapsack Capacity (M)  : {capacity}")
        print(f"Total Objects          : {len(weights)} (1 heavy + {2 * target_ratio} light)")
        print(f"Greedy Solution F(I)   : {f_greedy} (selected {len(greedy_chosen)} item: indices {greedy_chosen})")
        print(f"Optimal Solution F*(I) : {f_optimal} (selected {len(opt_chosen)} items)")
        print(f"Observed Ratio F*(I)/F(I): {actual_ratio:.2f}")

    assert actual_ratio == target_ratio, f"Ratio mismatch: expected {target_ratio}, got {actual_ratio}"

    return {
        "target_ratio": target_ratio,
        "capacity": capacity,
        "greedy_value": f_greedy,
        "optimal_value": f_optimal,
        "actual_ratio": actual_ratio,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knapsack Greedy Ratio Parameterized Analysis")
    parser.add_argument("--ratio", type=int, default=None, help="Input positive integer ratio R")
    args = parser.parse_args()

    print("=" * 85)
    print("Assignment 7: Analysis of Greedy-Knapsack by Profit Value Order")
    print("=" * 85)
    print("QUESTION: How large can the ratio F*(I) / F(I) get?")
    print("ANSWER  : The ratio can grow ARBITRARILY LARGE (UNBOUNDED, i.e., F*(I)/F(I) -> infinity).")
    print("REASON  : Processing by profit alone ignores item weights. A single item with slightly")
    print("          higher profit can consume the entire knapsack capacity, preventing an arbitrary")
    print("          number of lightweight, high-density items from being selected.")
    print("=" * 85)

    if args.ratio is not None:
        analyze_greedy_knapsack_ratio(args.ratio, verbose=True)
    else:
        # Demonstrate across a suite of positive integer ratios
        test_ratios = [1, 2, 4, 7, 10, 25, 50]
        print(f"Demonstrating parameterized instances for ratios: {test_ratios}")
        print("-" * 85)
        print(f"{'Target Ratio R':<16} | {'Knapsack Cap':<14} | {'Greedy F(I)':<14} | {'Optimal F*(I)':<14} | {'F*(I) / F(I)':<12}")
        print("-" * 85)

        for r in test_ratios:
            res = analyze_greedy_knapsack_ratio(r, verbose=False)
            print(f"{res['target_ratio']:<16} | {res['capacity']:<14} | {res['greedy_value']:<14} | {res['optimal_value']:<14} | {res['actual_ratio']:<12.1f}")

        print("-" * 85)
        print("All parameterized instances verified! F*(I) / F(I) strictly equals R for every R >= 1.")
        print("=" * 85)
