"""
Assignment 2:
Suppose we have a set of activities to schedule across a large number of lecture halls.
We wish to schedule all activities using as few lecture halls as possible.
Give an efficient greedy algorithm to determine which activity should use which lecture hall
and implement this algorithm.
"""

import heapq
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.lab_8.interval import Activity, verify_schedule_validity


def schedule_lecture_halls(
    activities: List[Activity],
) -> Tuple[int, Dict[int, List[Activity]], Dict[int, int]]:
    """
    Schedules activities into the minimum number of lecture halls such that no two overlapping
    activities share the same hall (Interval Partitioning Problem).

    Parameters:
        activities (List[Activity]): List of activities with start and finish times.

    Returns:
        Tuple[int, Dict[int, List[Activity]], Dict[int, int]]:
            - total_halls (int): Minimum number of lecture halls required.
            - hall_schedule (Dict[int, List[Activity]]): Mapping from hall_id to list of scheduled activities.
            - activity_hall_map (Dict[int, int]): Mapping from activity_id to assigned hall_id.

    Greedy Strategy:
        1. Sort all activities by start time: s_1 <= s_2 <= ... <= s_n.
        2. Maintain a min-heap of lecture halls currently in use, keyed by the finish time
           of the last activity assigned to that hall: (free_time, hall_id).
        3. For each activity in sorted order:
           a. If min-heap is non-empty and root.free_time <= activity.start:
              - Hall is free! Reuse this hall. Pop it, assign the activity, and push (activity.finish, hall_id).
           b. Else:
              - All current halls are busy at activity.start. Allocate a new lecture hall.
              - Assign the activity and push (activity.finish, new_hall_id).

    Optimality Justification:
        Let 'depth' d be the maximum number of mutually overlapping activities at any instant in time.
        Every valid schedule requires at least d lecture halls.
        The greedy algorithm never opens a new hall unless all existing halls are simultaneously
        occupied at the start time of the current activity (demonstrating an overlap depth equal
        to the number of allocated halls). Thus, it never allocates more than d halls.
        Therefore, the number of lecture halls used is guaranteed to be minimal.

    Complexity:
        - Sorting: O(n log n)
        - Min-Heap operations: n operations on a heap of size at most d (where d <= n), taking O(n log d).
        Total Time Complexity: O(n log n).
        Total Space Complexity: O(n).
    """
    if not activities:
        return 0, {}, {}

    # Sort activities by start time; on tie, by finish time
    sorted_activities = sorted(activities, key=lambda a: (a.start, a.finish))

    # Min-heap storing tuples: (free_time, hall_id)
    hall_heap: List[Tuple[float, int]] = []
    hall_schedules: Dict[int, List[Activity]] = defaultdict(list)
    activity_hall_map: Dict[int, int] = {}
    next_hall_id = 1

    for act in sorted_activities:
        if hall_heap and hall_heap[0][0] <= act.start:
            # Reuse the hall that becomes free earliest
            free_time, hall_id = heapq.heappop(hall_heap)
            hall_schedules[hall_id].append(act)
            activity_hall_map[act.id] = hall_id
            heapq.heappush(hall_heap, (act.finish, hall_id))
        else:
            # All halls are occupied; allocate a new hall
            hall_id = next_hall_id
            next_hall_id += 1
            hall_schedules[hall_id].append(act)
            activity_hall_map[act.id] = hall_id
            heapq.heappush(hall_heap, (act.finish, hall_id))

    total_halls = len(hall_schedules)
    return total_halls, dict(hall_schedules), activity_hall_map


if __name__ == "__main__":
    print("=" * 75)
    print("Assignment 2: Lecture Hall Scheduling (Interval Partitioning)")
    print("=" * 75)

    # Sample dataset of lecture sessions
    sample_activities = [
        Activity(id=1, start=9.0, finish=10.5, name="Algorithms"),
        Activity(id=2, start=9.5, finish=11.0, name="Machine Learning"),
        Activity(id=3, start=10.0, finish=12.0, name="Database Systems"),
        Activity(id=4, start=11.0, finish=12.5, name="Computer Networks"),
        Activity(id=5, start=12.0, finish=13.5, name="Operating Systems"),
        Activity(id=6, start=13.0, finish=14.5, name="Software Engineering"),
        Activity(id=7, start=14.0, finish=16.0, name="Artificial Intelligence"),
        Activity(id=8, start=15.0, finish=17.0, name="Computer Architecture"),
        Activity(id=9, start=11.5, finish=14.0, name="Cloud Computing"),
        Activity(id=10, start=13.5, finish=15.5, name="Cybersecurity"),
    ]

    print(f"Total Activities to Schedule: {len(sample_activities)}\n")
    print(f"{'ID':<4} | {'Activity Name':<24} | {'Start':<6} | {'Finish':<6} | {'Duration'}")
    print("-" * 55)
    for act in sample_activities:
        print(f"{act.id:<4} | {act.name:<24} | {act.start:<6.1f} | {act.finish:<6.1f} | {act.duration:.1f} hrs")
    print("-" * 55)

    total_halls, schedules, mapping = schedule_lecture_halls(sample_activities)

    print(f"\nMinimum Lecture Halls Required: {total_halls}\n")
    print("Detailed Hall-by-Hall Schedule:")
    print("=" * 75)
    for hall_id in sorted(schedules.keys()):
        acts = schedules[hall_id]
        print(f"Lecture Hall #{hall_id}: ({len(acts)} activities assigned)")
        for act in acts:
            print(f"   [{act.start:04.1f} - {act.finish:04.1f}] : {act.name} (ID: {act.id})")
        print()

    # Verification: Ensure no two overlapping activities are in the same hall
    is_valid = verify_schedule_validity(schedules)
    assert is_valid, "Invalid schedule detected! Overlapping activities in same hall."
    print("=" * 75)
    print("Verification Passed: No overlapping activities share any lecture hall!")
    print("=" * 75)
