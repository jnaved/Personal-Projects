"""
========================================================================================
DAA Lab 12: Job Assignment Problem using Branch and Bound
========================================================================================

Problem Statement:
------------------
A company has 4 employees (0, 1, 2, 3) and 4 jobs (0, 1, 2, 3).
Each employee must receive exactly one job, and each job must be assigned to
exactly one employee.
The cost matrix C is:

        Job 0   Job 1   Job 2   Job 3
Emp 0 [   9       2       7       8   ]
Emp 1 [   6       4       3       7   ]
Emp 2 [   5       8       1       8   ]
Emp 3 [   7       6       9       4   ]

Find the minimum-cost assignment using Branch and Bound.

----------------------------------------------------------------------------------------
1. Design of the Lower-Bound Function:
----------------------------------------------------------------------------------------
For a partially completed assignment node where employees 0 to (worker - 1) have
already been assigned jobs:
  - cost: Exact sum of costs for already assigned employees.
  - available_jobs: Set of jobs not yet assigned.
  - unassigned_employees: Employees from 'worker' to (n - 1).

Lower Bound Formula:
  LB = cost + sum_{i = worker}^{n-1} ( min_{j in available_jobs} C[i][j] )

Why is this admissible?
  In any valid completion, each remaining employee must be assigned one unique
  job from available_jobs. Since the actual assigned cost C[i][job] >= min_{j} C[i][j],
  the formula never overestimates the true completion cost (LB <= True Cost).

Bounding / Pruning Rule:
  Maintain 'best_cost' (cost of the best complete assignment found so far).
  If a node's LB >= best_cost, the node is discarded/pruned because no completion
  through this branch can strictly improve the solution.

----------------------------------------------------------------------------------------
2. Explanation: Least Lower Bound (LCBB) vs Expanding in Order of Generation (FIFO):
----------------------------------------------------------------------------------------
Question:
"Explain whether choosing the next node using the least lower bound can reduce the
search compared with expanding nodes in the order they are generated."

Answer:
YES, choosing the next node using the least lower bound (LCBB / Best-First Search)
dramatically reduces the search compared to expanding nodes in the order they are
generated (FIFO / Breadth-First Search).

Why LCBB reduces the search:
1. Guided Exploration:
   - FIFO (Queue / BFS) blindly expands all nodes level-by-level without regard to cost.
   - LCBB (Priority Queue / Min-Heap) always expands the live node with the smallest LB,
     moving greedily towards the most promising regions of the search space.

2. Early Discovery of a Tight Upper Bound:
   - In FIFO, because nodes are expanded level-by-level, no leaf (complete assignment)
     is reached until the search traverses all of Level 1, Level 2, and Level 3.
     Thus, 'best_cost' remains infinity throughout the entire upper tree, meaning
     FIFO CANNOT PRUNE ANY NODE at levels 1 or 2!
   - In LCBB, the algorithm descends directly to an optimal leaf node on its 4th
     expansion, immediately setting best_cost = 13.

3. Massive Subtree Pruning:
   - Once best_cost = 13 is found early, all other branches in the priority queue
     with LB >= 13 are pruned immediately upon extraction.
   - For example, at Level 1:
     * Node (Emp 0 -> Job 0) has LB = 17 >= 13 -> PRUNED (entire subtree skipped!)
     * Node (Emp 0 -> Job 2) has LB = 20 >= 13 -> PRUNED (entire subtree skipped!)
     * Node (Emp 0 -> Job 3) has LB = 18 >= 13 -> PRUNED (entire subtree skipped!)
   - In contrast, FIFO expands every single one of these non-promising subtrees.

Empirical Comparison on this 4x4 Matrix:
  +------------------+-------------------------+-----------------------+
  | Metric           | FIFO (Generation Order) | LCBB (Least LB)       |
  +------------------+-------------------------+-----------------------+
  | Nodes Generated  | 65                      | 11  (83.1% reduction) |
  | Nodes Expanded   | 41                      | 4   (90.2% reduction) |
  | Nodes Pruned     | 22                      | 6                     |
  | Optimal Cost     | 13                      | 13                    |
  | Assignment       | [1, 0, 2, 3]            | [1, 0, 2, 3]          |
  +------------------+-------------------------+-----------------------+
========================================================================================
"""

import heapq
from collections import deque

# Given Cost Matrix C
C = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
]
N = len(C)


def get_lower_bound(cost, worker, assigned_jobs):
    """Calculates admissible lower bound for a partial assignment."""
    available_jobs = set(range(N)) - set(assigned_jobs)
    lb = cost
    for w in range(worker, N):
        lb += min(C[w][j] for j in available_jobs)
    return lb


def solve_lcbb():
    """Least-Cost Branch and Bound (Priority Queue / Min-Heap)."""
    # Min-heap stores: (lb, counter, worker, cost, assigned_jobs)
    pq = []
    counter = 0
    root_lb = get_lower_bound(0, 0, ())
    heapq.heappush(pq, (root_lb, counter, 0, 0, ()))

    best_cost = float('inf')
    best_assignment = None
    nodes_generated, nodes_expanded, nodes_pruned = 1, 0, 0

    while pq:
        lb, _, worker, cost, assigned = heapq.heappop(pq)

        # Prune if LB >= best known complete solution cost
        if lb >= best_cost:
            nodes_pruned += 1
            continue

        # Complete assignment reached
        if worker == N:
            if cost < best_cost:
                best_cost = cost
                best_assignment = assigned
            continue

        nodes_expanded += 1
        available_jobs = [j for j in range(N) if j not in assigned]

        for job in available_jobs:
            counter += 1
            nodes_generated += 1
            child_cost = cost + C[worker][job]
            child_assigned = assigned + (job,)
            child_lb = get_lower_bound(child_cost, worker + 1, child_assigned)

            if child_lb >= best_cost:
                nodes_pruned += 1
            else:
                heapq.heappush(pq, (child_lb, counter, worker + 1, child_cost, child_assigned))

    return best_cost, best_assignment, nodes_generated, nodes_expanded, nodes_pruned


def solve_fifo():
    """FIFO Branch and Bound (Queue / Breadth-First Search)."""
    # Queue stores: (lb, worker, cost, assigned_jobs)
    queue = deque([(get_lower_bound(0, 0, ()), 0, 0, ())])

    best_cost = float('inf')
    best_assignment = None
    nodes_generated, nodes_expanded, nodes_pruned = 1, 0, 0

    while queue:
        lb, worker, cost, assigned = queue.popleft()

        if lb >= best_cost:
            nodes_pruned += 1
            continue

        if worker == N:
            if cost < best_cost:
                best_cost = cost
                best_assignment = assigned
            continue

        nodes_expanded += 1
        available_jobs = [j for j in range(N) if j not in assigned]

        for job in available_jobs:
            nodes_generated += 1
            child_cost = cost + C[worker][job]
            child_assigned = assigned + (job,)
            child_lb = get_lower_bound(child_cost, worker + 1, child_assigned)

            if child_lb >= best_cost:
                nodes_pruned += 1
            else:
                queue.append((child_lb, worker + 1, child_cost, child_assigned))

    return best_cost, best_assignment, nodes_generated, nodes_expanded, nodes_pruned


if __name__ == "__main__":
    print("--- DAA Lab 12: Job Assignment Problem ---")

    # 1. Run LCBB (Least Lower Bound)
    lcbb_cost, lcbb_assign, lcbb_gen, lcbb_exp, lcbb_prune = solve_lcbb()

    # 2. Run FIFO (Generation Order)
    fifo_cost, fifo_assign, fifo_gen, fifo_exp, fifo_prune = solve_fifo()

    # 3. Print Results & Comparison
    print(f"\nOptimal Assignment Cost : {lcbb_cost}")
    print(f"Optimal Assignment      : {lcbb_assign}")
    for emp, job in enumerate(lcbb_assign):
        print(f"  Employee {emp} -> Job {job} (Cost = {C[emp][job]})")

    print("\nSearch Comparison (LCBB vs FIFO):")
    print(f"{'Metric':<20} | {'FIFO (Order of Gen)':<20} | {'LCBB (Least LB)':<18}")
    print("-" * 64)
    print(f"{'Nodes Generated':<20} | {fifo_gen:<20} | {lcbb_gen:<18}")
    print(f"{'Nodes Expanded':<20} | {fifo_exp:<20} | {lcbb_exp:<18}")
    print(f"{'Nodes Pruned':<20} | {fifo_prune:<20} | {lcbb_prune:<18}")
    print("-" * 64)
    print(f"LCBB reduced node expansions by {((fifo_exp - lcbb_exp) / fifo_exp) * 100:.1f}%\n")
