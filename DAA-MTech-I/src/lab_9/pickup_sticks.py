"""
Assignment 3:
You are given a set of N sticks, each specified by its two endpoints as ordered triples (x, y, z).
No stick is vertical, and a stick can only be picked up if no other sticks are lying on top of it.
(a) Write a routine that takes two sticks a and b and reports whether a is above, below, or
unrelated to b.
(b) Give an algorithm that determines whether it is possible to pick up all the sticks, and if so,
provides a sequence of stick pickups that accomplishes this.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Point3D = Tuple[float, float, float]


@dataclass
class Stick:
    id: int
    p1: Point3D
    p2: Point3D

    def __repr__(self) -> str:
        return f"Stick({self.id})"


def check_stick_relation(a: Stick, b: Stick, eps: float = 1e-9) -> str:
    """
    Determines if stick a is 'above', 'below', or 'unrelated' to stick b.

    Parameters:
        a (Stick): First stick with endpoints p1 and p2.
        b (Stick): Second stick with endpoints p1 and p2.

    Returns:
        str: 'above', 'below', or 'unrelated'.
    """
    # 2D projection vectors on xy-plane
    dx_a = a.p2[0] - a.p1[0]
    dy_a = a.p2[1] - a.p1[1]

    dx_b = b.p2[0] - b.p1[0]
    dy_b = b.p2[1] - b.p1[1]

    # Cross product of direction vectors
    denom = dx_a * dy_b - dy_a * dx_b

    # Parallel or nearly parallel segments in 2D
    if abs(denom) < eps:
        return "unrelated"

    # Solve for intersection parameters t (for stick a) and u (for stick b)
    # a.p1 + t * (a.p2 - a.p1) == b.p1 + u * (b.p2 - b.p1)
    dx_ab = b.p1[0] - a.p1[0]
    dy_ab = b.p1[1] - a.p1[1]

    t = (dx_ab * dy_b - dy_ab * dx_b) / denom
    u = (dx_ab * dy_a - dy_ab * dx_a) / denom

    # Check if the intersection point lies strictly within both line segments
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        t_clamped = max(0.0, min(1.0, t))
        u_clamped = max(0.0, min(1.0, u))

        # Calculate z-height of both sticks at the 2D intersection point
        z_a = a.p1[2] + t_clamped * (a.p2[2] - a.p1[2])
        z_b = b.p1[2] + u_clamped * (b.p2[2] - b.p1[2])

        if z_a > z_b + eps:
            return "above"
        elif z_a < z_b - eps:
            return "below"
        else:
            return "unrelated"

    return "unrelated"


def find_pickup_sequence(sticks: List[Stick]) -> Tuple[bool, Optional[List[int]], str]:
    """
    Determines if all sticks can be picked up, and if so, returns a valid pickup order.

    Algorithm:
        1. Build a directed dependency graph:
           If stick a is 'above' stick b, stick a lies on top of b.
           Therefore, stick a must be picked up before stick b (edge: a -> b).
           The in-degree of b increments by 1.
        2. Perform Topological Sort (Kahn's Algorithm):
           - Collect all sticks with in-degree == 0 (no sticks resting on them).
           - Remove sticks one by one, decrementing the in-degrees of sticks below them.
        3. If all N sticks are processed, return the valid order.
           If a cycle exists, return False (impossible to pick up all sticks).

    Complexity:
        Pairwise relation test: O(N^2)
        Topological Sort: O(N + E) where E <= N^2
        Total Time Complexity: O(N^2)
        Total Space Complexity: O(N^2)
    """
    n = len(sticks)
    adj: Dict[int, List[int]] = defaultdict(list)
    in_degree: Dict[int, int] = {s.id: 0 for s in sticks}

    # Pairwise comparison of all sticks
    for i in range(n):
        for j in range(i + 1, n):
            s_i, s_j = sticks[i], sticks[j]
            relation = check_stick_relation(s_i, s_j)

            if relation == "above":
                # s_i is on top of s_j -> s_i must be picked up before s_j
                adj[s_i.id].append(s_j.id)
                in_degree[s_j.id] += 1
            elif relation == "below":
                # s_j is on top of s_i -> s_j must be picked up before s_i
                adj[s_j.id].append(s_i.id)
                in_degree[s_i.id] += 1

    # Kahn's Algorithm for Topological Sort
    queue = deque([stick_id for stick_id, deg in in_degree.items() if deg == 0])
    pickup_order: List[int] = []

    while queue:
        curr = queue.popleft()
        pickup_order.append(curr)

        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(pickup_order) == n:
        return True, pickup_order, "Valid pickup sequence found."
    else:
        return False, None, "Cycle detected: Impossible to pick up all sticks (deadlock)."


if __name__ == "__main__":
    print("=" * 75)
    print("Assignment 3: 3D Stick Pickup Routine & Topological Scheduling")
    print("=" * 75)

    # Define 4 sticks in 3D space
    # Stick 1 lies on top of Stick 2
    # Stick 2 lies on top of Stick 3
    # Stick 4 is unrelated to all others
    sample_sticks = [
        Stick(id=1, p1=(0, 2, 5), p2=(4, 2, 5)),       # Horizontal at y=2, z=5 (Top)
        Stick(id=2, p1=(2, 0, 3), p2=(2, 4, 3)),       # Vertical in xy at x=2, z=3 (Middle)
        Stick(id=3, p1=(0, 2, 1), p2=(4, 2, 1)),       # Horizontal at y=2, z=1 (Bottom)
        Stick(id=4, p1=(10, 10, 2), p2=(14, 10, 2)),   # Far away at x=10..14 (Unrelated)
    ]

    print("--- (a) Pairwise Relation Routine Demonstration ---")
    rel_1_2 = check_stick_relation(sample_sticks[0], sample_sticks[1])
    rel_2_3 = check_stick_relation(sample_sticks[1], sample_sticks[2])
    rel_1_4 = check_stick_relation(sample_sticks[0], sample_sticks[3])

    print(f"Relation(Stick 1, Stick 2): {rel_1_2} (Expected: above)")
    print(f"Relation(Stick 2, Stick 3): {rel_2_3} (Expected: above)")
    print(f"Relation(Stick 1, Stick 4): {rel_1_4} (Expected: unrelated)")
    assert rel_1_2 == "above"
    assert rel_2_3 == "above"
    assert rel_1_4 == "unrelated"

    print("\n--- (b) Pickup Sequence Determination ---")
    success, order, msg = find_pickup_sequence(sample_sticks)
    print(f"Status        : {msg}")
    print(f"Pickup Order  : {order}")
    assert success is True
    # Stick 1 must precede 2, which must precede 3
    idx1 = order.index(1)
    idx2 = order.index(2)
    idx3 = order.index(3)
    assert idx1 < idx2 < idx3
    print("Verification Passed: Valid topological stick pickup sequence generated!")
    print("=" * 75)
