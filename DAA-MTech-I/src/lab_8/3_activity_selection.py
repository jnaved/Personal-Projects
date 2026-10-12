"""
Assignment 3:
This concerns the first activity selection problem discussed in class to select a maximal set of
mutually compatible activities from a given set of activities with defined start and finish times.
Suppose that instead of selecting the first activity to finish, we select the last activity to start
that is compatible with all the previous activities. Write a program that implements this
algorithm and analyze its time complexity. Also, write a program that implements the optimal
algorithm discussed in class and analyze its time complexity. Compare the time complexities
of the two different programs that you have implemented.
"""

import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.lab_8.interval import Activity


# ==============================================================================
# 1. OPTIMAL ALGORITHM DISCUSSED IN CLASS: EARLIEST FINISH TIME (EFT)
# ==============================================================================
def activity_selection_earliest_finish(activities: List[Activity]) -> List[Activity]:
    """
    Classic Greedy Algorithm for Activity Selection (Earliest Finish Time - EFT).

    Strategy:
        1. Sort activities in non-decreasing order of their finish times:
           f_1 <= f_2 <= ... <= f_n.
        2. Pick the first activity (which finishes earliest, leaving maximum remaining time).
        3. Scan forward: select the next activity whose start time >= finish time of the
           last selected activity.

    Time Complexity Analysis:
        - Sorting: O(n log n) using standard comparison sort.
        - Linear Scan: O(n) single pass to check compatibility.
        - Total Time: O(n log n).
        - Auxiliary Space: O(n) to store sorted activities and selected result.

    Optimality:
        Provably optimal via greedy exchange argument. It guarantees maximum cardinality.
    """
    if not activities:
        return []

    # Sort primarily by finish time ascending, secondarily by start time
    sorted_acts = sorted(activities, key=lambda a: (a.finish, a.start))

    selected = [sorted_acts[0]]
    last_finish = sorted_acts[0].finish

    for act in sorted_acts[1:]:
        if act.start >= last_finish:
            selected.append(act)
            last_finish = act.finish

    return selected


# ==============================================================================
# 2. VARIANT ALGORITHM: LATEST START TIME (LST - BACKWARD GREEDY)
# ==============================================================================
def activity_selection_latest_start(activities: List[Activity]) -> List[Activity]:
    """
    Variant Greedy Algorithm: Select the latest activity to start that is compatible
    with previously selected activities (Latest Start Time - LST).

    Strategy:
        1. Sort activities in non-increasing order of their start times:
           s_1 >= s_2 >= ... >= s_n.
        2. Pick the first activity in this order (the activity that starts latest).
        3. Scan backward in time: select the next activity whose finish time <= start time
           of the last selected activity.

    Time Complexity Analysis:
        - Sorting: O(n log n) using standard comparison sort.
        - Linear Scan: O(n) single pass backward in time.
        - Total Time: O(n log n).
        - Auxiliary Space: O(n) to store sorted activities and selected result.

    Optimality:
        By time-reversal symmetry (t -> -t), selecting the latest starting compatible activity
        is the exact mathematical dual of the earliest finish time algorithm.
        It also provably yields a maximum-cardinality set of mutually compatible activities.
    """
    if not activities:
        return []

    # Sort primarily by start time descending, secondarily by finish time descending
    sorted_acts = sorted(activities, key=lambda a: (a.start, a.finish), reverse=True)

    selected = [sorted_acts[0]]
    last_start = sorted_acts[0].start

    for act in sorted_acts[1:]:
        if act.finish <= last_start:
            selected.append(act)
            last_start = act.start

    # Reverse so the final list is in chronological order
    selected.reverse()
    return selected


# ==============================================================================
# COMPLEXITY ANALYSIS & EMPIRICAL COMPARISON
# ==============================================================================
def compare_algorithms():
    """
    Benchmarks both algorithms on various input sizes, verifies their optimality,
    and compares their execution times.
    """
    print("=" * 85)
    print("Assignment 3: Activity Selection - Earliest Finish (EFT) vs. Latest Start (LST)")
    print("=" * 85)

    # 1. Verification on a known example
    toy_activities = [
        Activity(id=1, start=1, finish=4, name="A1"),
        Activity(id=2, start=3, finish=5, name="A2"),
        Activity(id=3, start=0, finish=6, name="A3"),
        Activity(id=4, start=5, finish=7, name="A4"),
        Activity(id=5, start=3, finish=9, name="A5"),
        Activity(id=6, start=5, finish=9, name="A6"),
        Activity(id=7, start=6, finish=10, name="A7"),
        Activity(id=8, start=8, finish=11, name="A8"),
        Activity(id=9, start=8, finish=12, name="A9"),
        Activity(id=10, start=2, finish=14, name="A10"),
        Activity(id=11, start=12, finish=16, name="A11"),
    ]

    res_eft = activity_selection_earliest_finish(toy_activities)
    res_lst = activity_selection_latest_start(toy_activities)

    print(f"Sample Instance ({len(toy_activities)} activities):")
    print(f"EFT Selected ({len(res_eft)} activities): {[a.name for a in res_eft]}")
    print(f"LST Selected ({len(res_lst)} activities): {[a.name for a in res_lst]}")
    assert len(res_eft) == len(res_lst) == 4, "Optimality check failed on sample instance"
    print("Both algorithms achieved the maximum optimal size of 4 activities!\n")

    # 2. Empirical Benchmark across problem sizes
    sizes = [1000, 5000, 20000, 50000, 100000]
    rng = random.Random(42)

    print("-" * 85)
    print(f"{'Size (N)':<10} | {'EFT Selected':<14} | {'LST Selected':<14} | {'EFT Time (ms)':<16} | {'LST Time (ms)':<16}")
    print("-" * 85)

    for n in sizes:
        # Generate random activities
        acts = []
        for i in range(n):
            s = rng.uniform(0, 1000)
            dur = rng.uniform(0.5, 50)
            acts.append(Activity(id=i + 1, start=s, finish=s + dur))

        # Benchmark Earliest Finish Time (Class algorithm)
        t0 = time.perf_counter()
        eft_sel = activity_selection_earliest_finish(acts)
        t_eft = (time.perf_counter() - t0) * 1000

        # Benchmark Latest Start Time (Requested variant)
        t0 = time.perf_counter()
        lst_sel = activity_selection_latest_start(acts)
        t_lst = (time.perf_counter() - t0) * 1000

        # Verification of optimal size match
        assert len(eft_sel) == len(lst_sel), f"Size mismatch at N={n}: {len(eft_sel)} != {len(lst_sel)}"

        print(f"{n:<10} | {len(eft_sel):<14} | {len(lst_sel):<14} | {t_eft:<16.3f} | {t_lst:<16.3f}")

    print("-" * 85)
    print("\nSummary & Comparison of Time Complexities:")
    print("1. Earliest Finish Time (EFT):")
    print("   - Sort activities by finish time ascending: O(n log n)")
    print("   - Single forward scan: O(n)")
    print("   - Total Time Complexity: O(n log n)")
    print("2. Latest Start Time (LST):")
    print("   - Sort activities by start time descending: O(n log n)")
    print("   - Single backward scan: O(n)")
    print("   - Total Time Complexity: O(n log n)")
    print("3. Asymptotic & Practical Comparison:")
    print("   - Both algorithms have identical asymptotic time complexity: O(n log n).")
    print("   - Both algorithms have O(n) auxiliary space complexity.")
    print("   - Both algorithms produce provably optimal maximal sets of compatible activities.")
    print("   - Practical execution times are virtually identical, governed by Python's Timsort.")
    print("=" * 85)


if __name__ == "__main__":
    compare_algorithms()
