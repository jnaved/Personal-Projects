"""
Constrained Sum of Subsets Problem using Backtracking
======================================================
DAA Lab 8.1 - Question 2.1

Problem Description:
--------------------
Given a set of n positive integers S and a target sum T, find all subsets
whose elements sum up exactly to T while satisfying:
  1. Cardinality constraints: subset size k must satisfy k_min <= k <= k_max.
  2. Mandatory elements: must appear in every valid subset.
  3. Excluded elements: cannot appear in any valid subset.
  4. Duplicate distinction: distinguishes element positions even when values match.

Pruning / Bounding Conditions:
------------------------------
  1. Sum Overflow: Prune if current_sum + element > T.
  2. Insufficient Remaining Sum: Prune if current_sum + remaining_eligible_sum < T.
  3. Max Cardinality Bound: Prune if current_size >= k_max and target is not reached.
  4. Min Cardinality Bound: Prune if current_size + remaining_eligible_count < k_min.
  5. Excluded Element: Prune branch attempting to include an excluded element.
  6. Mandatory Element: Prune branch attempting to omit a mandatory element.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple


# =====================================================================
# 1. INPUT VALIDATION (CONTRADICTION REJECTION)
# =====================================================================

def validate_inputs(
    numbers: List[int],
    target: int,
    k_min: int,
    k_max: int,
    mandatory: Set[int],
    excluded: Set[int]
) -> None:
    """
    Validates input parameters and rejects contradictory or impossible constraints.
    """
    # 1. Target and element values must be positive
    if target <= 0:
        raise ValueError(f"Contradictory Input: Target sum T must be positive (got {target}).")
    for x in numbers:
        if x <= 0:
            raise ValueError(f"Contradictory Input: All elements in S must be positive integers (found {x}).")

    # 2. Overlapping mandatory and excluded elements
    overlap = mandatory.intersection(excluded)
    if overlap:
        raise ValueError(f"Contradictory Input: Elements cannot be both mandatory and excluded: {overlap}")

    # 3. Cardinality bounds
    n = len(numbers)
    if k_min < 0 or k_max < k_min:
        raise ValueError(f"Contradictory Input: Invalid cardinality range [{k_min}, {k_max}].")
    if k_max > n:
        raise ValueError(f"Contradictory Input: k_max ({k_max}) cannot exceed total elements ({n}).")

    # 4. Mandatory elements must exist in the input set
    available_pool = list(numbers)
    for m in mandatory:
        if m in available_pool:
            available_pool.remove(m)
        else:
            raise ValueError(f"Contradictory Input: Mandatory element {m} is not present in S.")

    # 5. Check if mandatory elements already violate target or cardinality
    if sum(mandatory) > target:
        raise ValueError(f"Contradictory Input: Sum of mandatory elements ({sum(mandatory)}) exceeds target T ({target}).")
    if len(mandatory) > k_max:
        raise ValueError(f"Contradictory Input: Number of mandatory elements ({len(mandatory)}) exceeds k_max ({k_max}).")


# =====================================================================
# 2. CORE CONSTRAINED BACKTRACKING SOLVER
# =====================================================================

def constrained_sum_of_subsets(
    numbers: List[int],
    target: int,
    k_min: int,
    k_max: int,
    mandatory: Optional[Set[int]] = None,
    excluded: Optional[Set[int]] = None,
    verbose: bool = True
) -> Tuple[List[List[Tuple[int, int]]], Dict[str, Any], List[str]]:
    """
    Finds all valid subsets satisfying target sum, cardinality bounds,
    and mandatory/excluded elements.

    Parameters:
      numbers: List of positive integers (elements can have duplicates).
      target: Desired target sum T.
      k_min: Minimum permitted subset size.
      k_max: Maximum permitted subset size.
      mandatory: Set of values that MUST be included in valid subsets.
      excluded: Set of values that CANNOT be included in valid subsets.
      verbose: If True, records detailed reasons for every pruned branch.

    Returns:
      (valid_subsets, stats, pruning_log):
        - valid_subsets: List of matching subsets, each represented as
                         [(index_0, value_0), (index_1, value_1), ...]
        - stats: Summary of search metrics.
        - pruning_log: Detailed list of explanations for each pruned branch.
    """
    mandatory_set = set(mandatory or [])
    excluded_set = set(excluded or [])

    # Validate inputs upfront
    validate_inputs(numbers, target, k_min, k_max, mandatory_set, excluded_set)

    n = len(numbers)

    # Represent each element with its original 0-indexed position: (index, value)
    # This ensures duplicates at different positions are clearly distinguished.
    items: List[Tuple[int, int]] = [(i, numbers[i]) for i in range(n)]

    # Suffix sums and counts of eligible (non-excluded) elements for O(1) remaining bound checks
    eligible_suffix_sum = [0] * (n + 1)
    eligible_suffix_count = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        idx, val = items[i]
        is_eligible = val not in excluded_set
        eligible_suffix_sum[i] = eligible_suffix_sum[i + 1] + (val if is_eligible else 0)
        eligible_suffix_count[i] = eligible_suffix_count[i + 1] + (1 if is_eligible else 0)

    # Suffix counts of mandatory elements remaining after each index
    mandatory_suffix_count = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        _, val = items[i]
        mandatory_suffix_count[i] = mandatory_suffix_count[i + 1] + (1 if val in mandatory_set else 0)

    # Search metrics
    stats: Dict[str, Any] = {
        "nodes_explored": 0,
        "valid_subsets_count": 0,
        "pruned_sum_overflow": 0,
        "pruned_insufficient_sum": 0,
        "pruned_max_cardinality": 0,
        "pruned_min_cardinality": 0,
        "pruned_excluded": 0,
        "pruned_mandatory_omitted": 0,
        "elapsed_time_ms": 0.0,
    }

    valid_subsets: List[List[Tuple[int, int]]] = []
    pruning_log: List[str] = []
    current_subset: List[Tuple[int, int]] = []

    start_time = time.perf_counter()

    def format_path(subset: List[Tuple[int, int]]) -> str:
        if not subset:
            return "{}"
        return "{" + ", ".join(f"{v} (pos {i})" for i, v in subset) + "}"

    def backtrack(idx: int, curr_sum: int, mandatory_chosen: Set[int]) -> None:
        stats["nodes_explored"] += 1
        curr_size = len(current_subset)

        # -------------------------------------------------------------
        # 1. CHECK GOAL: Exact Target Reached
        # -------------------------------------------------------------
        if curr_sum == target:
            # Check cardinality bounds
            if curr_size < k_min:
                stats["pruned_min_cardinality"] += 1
                if verbose:
                    pruning_log.append(
                        f"Pruned at {format_path(current_subset)}: Sum reached {target}, "
                        f"but size {curr_size} < k_min ({k_min})."
                    )
                return

            # Check all mandatory elements are included
            if not mandatory_set.issubset(mandatory_chosen):
                stats["pruned_mandatory_omitted"] += 1
                if verbose:
                    missing = mandatory_set - mandatory_chosen
                    pruning_log.append(
                        f"Pruned at {format_path(current_subset)}: Sum reached {target}, "
                        f"but missing mandatory element(s): {missing}."
                    )
                return

            # Valid subset found!
            valid_subsets.append(list(current_subset))
            stats["valid_subsets_count"] += 1
            # Since all remaining elements are positive, extending further will exceed target
            return

        # Base case: reached end of items
        if idx >= n:
            return

        elem_idx, elem_val = items[idx]

        # =============================================================
        # BRANCH 1: INCLUDE items[idx]
        # =============================================================
        # Constraint Pruning: Excluded element
        if elem_val in excluded_set:
            stats["pruned_excluded"] += 1
            if verbose:
                pruning_log.append(
                    f"Pruned [Include {elem_val} (pos {elem_idx})] at {format_path(current_subset)}: "
                    f"Element {elem_val} is in EXCLUDED list."
                )
        # Cardinality Pruning: Max size exceeded
        elif curr_size + 1 > k_max:
            stats["pruned_max_cardinality"] += 1
            if verbose:
                pruning_log.append(
                    f"Pruned [Include {elem_val} (pos {elem_idx})] at {format_path(current_subset)}: "
                    f"Subset size would become {curr_size + 1} > k_max ({k_max})."
                )
        # Sum Pruning: Target overflow
        elif curr_sum + elem_val > target:
            stats["pruned_sum_overflow"] += 1
            if verbose:
                pruning_log.append(
                    f"Pruned [Include {elem_val} (pos {elem_idx})] at {format_path(current_subset)}: "
                    f"Sum would become {curr_sum + elem_val} > target T ({target})."
                )
        else:
            # Safe to include
            current_subset.append((elem_idx, elem_val))
            new_mand = set(mandatory_chosen)
            if elem_val in mandatory_set:
                new_mand.add(elem_val)

            backtrack(idx + 1, curr_sum + elem_val, new_mand)

            # Undo move (backtrack)
            current_subset.pop()

        # =============================================================
        # BRANCH 2: EXCLUDE items[idx]
        # =============================================================
        # Constraint Pruning: Mandatory element cannot be omitted
        if elem_val in mandatory_set and elem_val not in mandatory_chosen:
            # Check if another copy of this mandatory value exists further down
            # If not, omitting this item makes reaching all mandatory elements impossible
            has_future_copy = any(items[j][1] == elem_val for j in range(idx + 1, n))
            if not has_future_copy:
                stats["pruned_mandatory_omitted"] += 1
                if verbose:
                    pruning_log.append(
                        f"Pruned [Exclude {elem_val} (pos {elem_idx})] at {format_path(current_subset)}: "
                        f"Mandatory element {elem_val} cannot be omitted (no later copy available)."
                    )
                return

        # Bound 2: Insufficient remaining sum to reach T
        rem_sum = eligible_suffix_sum[idx + 1]
        if curr_sum + rem_sum < target:
            stats["pruned_insufficient_sum"] += 1
            if verbose:
                pruning_log.append(
                    f"Pruned [Exclude {elem_val} (pos {elem_idx})] at {format_path(current_subset)}: "
                    f"Current sum {curr_sum} + remaining eligible sum {rem_sum} = {curr_sum + rem_sum} < target T ({target})."
                )
            return

        # Bound 3: Cardinality pruning - cannot reach k_min even if we take all remaining
        rem_count = eligible_suffix_count[idx + 1]
        if curr_size + rem_count < k_min:
            stats["pruned_min_cardinality"] += 1
            if verbose:
                pruning_log.append(
                    f"Pruned [Exclude {elem_val} (pos {elem_idx})] at {format_path(current_subset)}: "
                    f"Current size {curr_size} + remaining eligible items {rem_count} = {curr_size + rem_count} < k_min ({k_min})."
                )
            return

        # Safe to explore exclusion branch
        backtrack(idx + 1, curr_sum, mandatory_chosen)

    # Initiate search
    backtrack(0, 0, set())
    stats["elapsed_time_ms"] = (time.perf_counter() - start_time) * 1000

    return valid_subsets, stats, pruning_log


# =====================================================================
# 3. HELPER TO DISPLAY SUBSETS
# =====================================================================

def format_subset_str(subset: List[Tuple[int, int]]) -> str:
    """Formats a subset showing both values and distinct positions."""
    vals = [v for _, v in subset]
    details = ", ".join(f"{v} (pos {i})" for i, v in subset)
    return f"Values: {vals}  |  Positions: [{details}]  |  Sum = {sum(vals)}  |  Size = {len(vals)}"


# =====================================================================
# 4. MAIN LAB ASSIGNMENT DEMONSTRATION
# =====================================================================

if __name__ == "__main__":
    print("=" * 76)
    print(" DAA Lab 8.1 - Question 2.1: Constrained Sum of Subsets Backtracking ")
    print("=" * 76)

    # -----------------------------------------------------------------
    # PRIMARY TEST CASE (As specified in Question 2.1):
    # S = {3, 5, 6, 7, 9, 11, 12}, T = 21
    # Minimum subset size: 2, Maximum subset size: 4
    # Mandatory element: 3, Excluded element: 7
    # -----------------------------------------------------------------
    S = [3, 5, 6, 7, 9, 11, 12]
    T = 21
    k_min = 2
    k_max = 4
    mandatory = {3}
    excluded = {7}

    print("\n[Input Parameters]")
    print(f"  Input Set S           : {S} (n = {len(S)})")
    print(f"  Target Sum T          : {T}")
    print(f"  Permitted Subset Size : [{k_min}, {k_max}]")
    print(f"  Mandatory Element(s)  : {mandatory}")
    print(f"  Excluded Element(s)   : {excluded}")

    subsets, stats, prunings = constrained_sum_of_subsets(
        numbers=S,
        target=T,
        k_min=k_min,
        k_max=k_max,
        mandatory=mandatory,
        excluded=excluded,
        verbose=True
    )

    print("\n" + "-" * 76)
    print("[1. Valid Subsets Found]")
    print(f"  Total Valid Subsets: {len(subsets)}")
    if subsets:
        for i, s in enumerate(subsets, 1):
            print(f"  Subset #{i}: {format_subset_str(s)}")
    else:
        print("  No valid subsets satisfy all constraints.")

    print("\n" + "-" * 76)
    print("[2. Search & Pruning Statistics]")
    print(f"  State Nodes Explored              : {stats['nodes_explored']}")
    print(f"  Pruned (Sum Exceeded > T)         : {stats['pruned_sum_overflow']}")
    print(f"  Pruned (Insufficient Rem. Sum < T): {stats['pruned_insufficient_sum']}")
    print(f"  Pruned (Max Size Exceeded > k_max): {stats['pruned_max_cardinality']}")
    print(f"  Pruned (Min Size Unreachable)     : {stats['pruned_min_cardinality']}")
    print(f"  Pruned (Excluded Element Filter)  : {stats['pruned_excluded']}")
    print(f"  Pruned (Mandatory Omission Filter): {stats['pruned_mandatory_omitted']}")
    print(f"  Execution Time                    : {stats['elapsed_time_ms']:.3f} ms")

    print("\n" + "-" * 76)
    print("[3. Explanation of Why Rejected Branches Were Pruned]")
    for i, reason in enumerate(prunings, 1):
        print(f"  ({i:2d}) {reason}")

    # Assert correctness for Question 2.1
    # Only {3, 6, 12} satisfies: 3 + 6 + 12 = 21, size 3 in [2, 4], has 3, no 7.
    assert len(subsets) == 1
    assert [v for _, v in subsets[0]] == [3, 6, 12]
    print("\n  [VERIFICATION PASSED]: Exactly 1 valid subset {3, 6, 12} satisfies all constraints.")

    # -----------------------------------------------------------------
    # TEST CASE 2: Duplicate Values Distinction by Position
    # S = [2, 2, 4], T = 4, k_min = 1, k_max = 2
    # Subsets: [4 (pos 2)] (size 1) and [2 (pos 0), 2 (pos 1)] (size 2)
    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[Test Case 2: Duplicate Handling & Position Distinction]")
    S_dup = [2, 2, 4]
    T_dup = 4
    subsets_dup, _, _ = constrained_sum_of_subsets(
        numbers=S_dup,
        target=T_dup,
        k_min=1,
        k_max=2,
        verbose=False
    )
    print(f"  Set S = {S_dup}, Target T = {T_dup}, Size = [1, 2]")
    print(f"  Total Valid Subsets: {len(subsets_dup)}")
    for i, s in enumerate(subsets_dup, 1):
        print(f"  Subset #{i}: {format_subset_str(s)}")
    assert len(subsets_dup) == 2

    # -----------------------------------------------------------------
    # TEST CASE 3: Contradiction Rejection Validation
    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[Test Case 3: Contradiction Rejection Validation]")

    # 3a. Overlapping mandatory and excluded elements
    try:
        constrained_sum_of_subsets(S, target=21, k_min=2, k_max=4, mandatory={3, 7}, excluded={7})
        print("ERROR: Failed to reject overlapping mandatory/excluded elements!")
    except ValueError as e:
        print(f"  [PASSED] Rejection 1: {e}")

    # 3b. Mandatory elements sum exceeds target T
    try:
        constrained_sum_of_subsets(S, target=10, k_min=2, k_max=4, mandatory={11})
        print("ERROR: Failed to reject mandatory sum exceeding target!")
    except ValueError as e:
        print(f"  [PASSED] Rejection 2: {e}")

    # 3c. Invalid cardinality range
    try:
        constrained_sum_of_subsets(S, target=21, k_min=5, k_max=3)
        print("ERROR: Failed to reject k_min > k_max!")
    except ValueError as e:
        print(f"  [PASSED] Rejection 3: {e}")

    print("\n" + "=" * 76)
    print("  All Question 2.1 Requirements Successfully Tested & Verified!  ")
    print("=" * 76)
