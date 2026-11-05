"""
Assignment 5:
Consider the 0/1 knapsack problem with n objects, whose profits and weights are vi, wi (1 <= i <= n),
respectively. The capacity of the knapsack is m. It is also given that all the objects have
the same weight. Present an O(m) algorithm to solve and implement this problem.
"""

import random
from typing import Callable, List, Tuple, TypeVar

T = TypeVar("T")


def quickselect(arr: List[T], k: int, key: Callable[[T], any] = lambda x: x) -> T:
    """Finds the k-th smallest element (0-indexed) in arr in expected O(n) time."""
    if not (0 <= k < len(arr)):
        raise IndexError("k is out of bounds for quickselect.")

    left, right = 0, len(arr) - 1
    while left <= right:
        if left == right:
            return arr[left]

        pivot_val = key(arr[random.randint(left, right)])
        i, lt, gt = left, left, right
        while i <= gt:
            curr_val = key(arr[i])
            if curr_val < pivot_val:
                arr[lt], arr[i] = arr[i], arr[lt]
                lt += 1
                i += 1
            elif curr_val > pivot_val:
                arr[gt], arr[i] = arr[i], arr[gt]
                gt -= 1
            else:
                i += 1

        if k < lt:
            right = lt - 1
        elif k > gt:
            left = gt + 1
        else:
            return arr[k]

    return arr[k]


def get_top_k_largest(arr: List[T], k: int, key: Callable[[T], any] = lambda x: x) -> List[T]:
    """Returns the k largest elements in arr in O(n) expected time using Quickselect."""
    if k <= 0:
        return []
    if k >= len(arr):
        return list(arr)

    target_idx = len(arr) - k
    quickselect(arr, target_idx, key=key)
    return arr[target_idx:]


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
        1. Since every item consumes uniform_weight (w_0), the maximum number of items that
           can fit into capacity m is k = floor(m / uniform_weight).
        2. Maximizing total profit reduces strictly to selecting the k items with
           the largest profits.
        3. Using Quickselect, the top k elements can be found in O(n) = O(m) expected time.
        4. Summing the top k selected elements takes O(k) <= O(m) time.
        5. Total Time: O(m). Auxiliary Space: O(n).
    """
    n = len(profits)
    if n == 0 or capacity <= 0 or uniform_weight <= 0:
        return 0, []

    # Maximum number of items that can fit
    k = capacity // uniform_weight

    # Case 1: All items can fit in the knapsack
    if k >= n:
        return sum(profits), list(range(n))

    # Case 2: Knapsack can hold at most k items (k < n)
    indexed_profits = [(profits[i], i) for i in range(n)]
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
    print(f"Max Items Allowed (k)   : {k} (floor(10 / 2) = 5)")
    print(f"Item Profits            : {sample_profits}")

    # Solve using O(m) Quickselect algorithm
    opt_profit, chosen_idx = solve_knapsack_equal_weights(sample_profits, w_0, m)

    print("-" * 80)
    print(f"O(m) Algorithm Profit   : {opt_profit}")
    print(f"Chosen Item Indices     : {chosen_idx}")
    print(f"Chosen Profits          : {[sample_profits[i] for i in chosen_idx]}")

    # Verification: Top 5 profits are 30, 25, 20, 18, 15 -> sum = 108
    expected_profit = sum(sorted(sample_profits, reverse=True)[:k])
    assert opt_profit == expected_profit, f"Expected {expected_profit}, got {opt_profit}"
    print(f"Verification Passed: Correctly selected top {k} elements with profit {opt_profit}!")
    print("=" * 80)
