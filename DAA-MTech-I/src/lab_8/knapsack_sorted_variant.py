"""
Assignment 6:
Suppose in a 0-1 knapsack problem, the order of items sorted by increasing weight is the same
as their order when sorted by decreasing value. Give an efficient algorithm to find an optimal
solution to this variant of the knapsack problem and implement it.
"""

from typing import List, Tuple


def solve_sorted_variant_knapsack(
    weights: List[int],
    values: List[int],
    capacity: int,
) -> Tuple[int, List[int]]:
    """
    Solves the 0-1 knapsack problem variant where sorting by increasing weight
    yields the exact same order as sorting by decreasing value.

    Condition:
        w_1 <= w_2 <= ... <= w_n  AND  v_1 >= v_2 >= ... >= v_n.

    Parameters:
        weights (List[int]): Item weights w_i.
        values (List[int]): Item values / profits v_i.
        capacity (int): Knapsack capacity W.

    Returns:
        Tuple[int, List[int]]: (max_value, selected_original_indices)

    Algorithm (Greedy Strategy):
        1. Pair items with original indices: (weight, value, orig_index).
        2. Sort items such that weights are non-decreasing: w_1 <= w_2 <= ... <= w_n.
        3. Scan greedily from the first item:
           Include item i if weight fits in remaining capacity; stop once capacity is exceeded.

    Proof of Correctness (Exchange Argument):
        - Let G = {1, 2, ..., k} be the greedy prefix solution.
        - Suppose an optimal solution OPT differs from G.
        - Let i be the smallest index such that i in G and i not in OPT.
        - There must exist some item j in OPT not in G. Since G is a prefix {1, ..., k}, j > k >= i.
        - By sorting order: w_i <= w_j and v_i >= v_j.
        - Replacing item j with item i in OPT yields OPT' = (OPT \\ {j}) U {i}.
          - Weight(OPT') = Weight(OPT) - w_j + w_i <= Weight(OPT) <= capacity (feasible).
          - Value(OPT')  = Value(OPT)  - v_j + v_i >= Value(OPT) (at least as valuable).
        - Repeating this exchange transforms OPT into G without decreasing total value.
        - Therefore, the greedy solution G is strictly optimal.

    Complexity:
        - Sorting: O(n log n) (O(n) if already sorted).
        - Greedy selection: O(n) linear scan.
        - Total Time: O(n log n).
        - Auxiliary Space: O(n).
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0, []

    # Pair items: (weight, value, original_index)
    items = [(weights[i], values[i], i) for i in range(n)]
    items.sort(key=lambda item: (item[0], -item[1]))

    # Validate condition: verify values are non-increasing
    for i in range(len(items) - 1):
        if items[i][1] < items[i + 1][1]:
            raise ValueError(
                f"Input does not satisfy problem condition: "
                f"item at rank {i} (w={items[i][0]}, v={items[i][1]}) has lower value than "
                f"item at rank {i+1} (w={items[i+1][0]}, v={items[i+1][1]})."
            )

    # Greedy Selection: Pick items in order as long as they fit
    total_val = 0
    rem_cap = capacity
    selected_indices = []

    for w, v, orig_idx in items:
        if w <= rem_cap:
            selected_indices.append(orig_idx)
            total_val += v
            rem_cap -= w
        else:
            break

    return total_val, selected_indices


if __name__ == "__main__":
    print("=" * 80)
    print("Assignment 6: 0-1 Knapsack Variant (Increasing Weight == Decreasing Value)")
    print("=" * 80)

    # Sample instance where increasing weight corresponds to decreasing value:
    # Item 1: w=2,  v=100
    # Item 3: w=4,  v=80
    # Item 5: w=5,  v=70
    # Item 0: w=7,  v=50
    # Item 2: w=9,  v=30
    # Item 4: w=12, v=15
    sample_weights = [7, 2, 9, 4, 12, 5]
    sample_values = [50, 100, 30, 80, 15, 70]
    knapsack_cap = 15

    print(f"Knapsack Capacity : {knapsack_cap}")
    print(f"Weights           : {sample_weights}")
    print(f"Values            : {sample_values}\n")

    # Solve using greedy O(n log n) algorithm
    opt_val, chosen = solve_sorted_variant_knapsack(sample_weights, sample_values, knapsack_cap)

    print(f"Greedy Variant Optimal Value : {opt_val}")
    print(f"Selected Item Indices         : {chosen}")
    print("Selected Items Breakdown      :")
    for idx in chosen:
        print(f"   Item {idx}: weight={sample_weights[idx]}, value={sample_values[idx]}")
    total_w = sum(sample_weights[i] for i in chosen)
    print(f"Total Weight Used             : {total_w} / {knapsack_cap}")

    # Verification: items with w=2 (v=100), w=4 (v=80), w=5 (v=70) sum to 250 with weight 11 <= 15
    assert opt_val == 250, f"Expected 250, got {opt_val}"
    assert sorted(chosen) == [1, 3, 5], f"Expected items [1, 3, 5], got {chosen}"
    print("-" * 80)
    print("Verification Passed: Greedy solution achieves optimal value 250 with items [1, 3, 5]!")
    print("=" * 80)
