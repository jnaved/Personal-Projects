"""
========================================================================================
DAA Lab 12: Resource Budget Planning (0/1 Knapsack) using Branch and Bound
========================================================================================

Problem Statement:
------------------
A research team has a limited resource budget of W = 15 units and must select
a subset of indivisible projects to maximize total benefit.

Project    | P1 | P2 | P3 | P4 | P5 | P6
-----------+----+----+----+----+----+----
Benefit    | 12 | 10 | 20 | 15 |  8 | 14
Resource   |  4 |  3 |  7 |  5 |  2 |  6
Ratio(v/w) | 3.0| 3.3| 2.8| 3.0| 4.0| 2.3

Total available resource limit: W = 15 units.
Objective: Maximize sum_{i} (Benefit_i * x_i) subject to sum_{i} (Resource_i * x_i) <= 15,
           where x_i in {0, 1} (indivisible projects).

----------------------------------------------------------------------------------------
1. Why Fractional Selection is Useful for Upper Bounding (Continuous Relaxation):
----------------------------------------------------------------------------------------
Even though projects cannot be selected fractionally in reality (x_i in {0, 1}),
relaxing the integrality constraint to allow continuous fractions (0 <= x_i <= 1)
is extremely valuable because:

1. Admissible Upper Bound Guarantee (Relaxation Principle):
   The feasible region of the 0/1 integer problem is a strict subset of the fractional
   problem: {0, 1}^n subset [0, 1]^n.
   Expanding the search domain cannot decrease the maximum possible value.
   Therefore:
       Optimal_0/1_benefit <= Optimal_fractional_benefit
   The fractional knapsack solution serves as an admissible (optimistic) upper bound (UB).

2. Greedy O(n) Efficiency:
   The Fractional Knapsack Problem exhibits optimal substructure and can be solved
   greedily in linear time by simply taking items in descending order of benefit-to-resource
   ratio (v_i / w_i).

3. Tightness and Pruning Power:
   Because the continuous relaxation fills 100% of the remaining capacity using the highest
   density item available, the bound is very tight. Any node whose UB <= best_known_benefit
   can never produce a strictly better integer solution and is safely pruned.

----------------------------------------------------------------------------------------
2. How the Order of Projects Affects the Number of Nodes Explored:
----------------------------------------------------------------------------------------
The order in which projects are branched on significantly influences the state space tree:

1. Descending Benefit-to-Resource Ratio (v_i / w_i):
   - High-density projects (e.g. P5 with ratio 4.0, P2 with ratio 3.33) are branched on first.
   - Left branches (x_i = 1) rapidly accumulate high benefit with little resource consumption,
     quickly establishing a high incumbent lower bound ('best_benefit').
   - Exclude branches (x_i = 0) suffer a sharp drop in their fractional upper bound because
     they are denied the most efficient items, causing whole subtrees to be pruned early.

2. Arbitrary or Suboptimal Ordering:
   - Branching on low-density or large-resource items first delays finding high-quality
     feasible solutions.
   - The incumbent bound remains weak near the top of the tree, allowing non-promising
     subtrees to be needlessly expanded before they can be pruned.

----------------------------------------------------------------------------------------
3. Why a Node with Upper Bound EQUAL to Best Known Benefit Requires Careful Handling:
----------------------------------------------------------------------------------------
When a node's upper bound satisfies UB == best_known_benefit:

1. Single Optimum vs. All Alternate Optima:
   - If our goal is only to find A SINGLE optimal solution:
     The node CANNOT strictly exceed the incumbent (UB <= best_benefit).
     Thus, it can be pruned safely without losing optimality.
   - If our goal is to find ALL alternative optimal solutions:
     Pruning when UB == best_benefit will discard valid alternative subsets!
     In this exact problem, there are TWO distinct optimal subsets of benefit 45:
       Option A: {P2, P3, P4}       -> Resource = 3 + 7 + 5 = 15, Benefit = 10 + 20 + 15 = 45
       Option B: {P1, P2, P4, P5}   -> Resource = 4 + 3 + 5 + 2 = 14, Benefit = 12 + 10 + 15 + 8 = 45
     Pruning with UB <= 45 finds only one solution. Pruning strictly with UB < 45 discovers both!

2. Floating-Point Precision & Rounding Issues:
   - Fractional bounds involve division (e.g. v_k * rem_w / w_k).
   - In floating-point arithmetic, a theoretical bound of 45.0 might evaluate to 44.99999999999999.
   - Using a strict '>' or '<=' comparison without an epsilon tolerance (e.g. 1e-9) can cause
     a valid node to be mistakenly pruned due to round-off error.
========================================================================================
"""

import heapq

# Project definitions: (Name, Benefit, Resource Requirement)
PROJECTS = [
    ('P1', 12, 4),
    ('P2', 10, 3),
    ('P3', 20, 7),
    ('P4', 15, 5),
    ('P5', 8, 2),
    ('P6', 14, 6)
]
TOTAL_BUDGET = 15


def get_fractional_upper_bound(idx, current_val, current_weight, projects, budget=TOTAL_BUDGET):
    """
    Computes an admissible upper bound using the Fractional Knapsack continuous relaxation.
    Greedily fills remaining capacity using unconsidered projects in descending ratio order.
    """
    if current_weight > budget:
        return 0.0

    ub = float(current_val)
    remaining_capacity = budget - current_weight

    # Sort remaining projects by density (benefit / resource)
    remaining_projects = sorted(projects[idx:], key=lambda p: p[1] / p[2], reverse=True)

    for _, val, weight in remaining_projects:
        if weight <= remaining_capacity:
            remaining_capacity -= weight
            ub += val
        else:
            # Take fraction of the break project
            ub += val * (remaining_capacity / weight)
            break

    return ub


def solve_resource_planning(projects, budget=TOTAL_BUDGET, find_all_optima=False):
    """
    Solves Resource Budget Planning using Best-First Branch and Bound (Max-Heap by Upper Bound).
    
    Args:
        projects: List of (name, benefit, weight) tuples.
        budget: Maximum resource limit.
        find_all_optima: If False, prunes when UB <= best_val (faster, single optimum).
                         If True, prunes strictly when UB < best_val (finds all alternate optima).
    """
    # Max-heap stores: (-upper_bound, -current_val, counter, idx, current_val, current_weight, selected_tuple)
    pq = []
    counter = 0
    root_ub = get_fractional_upper_bound(0, 0, 0, projects, budget)
    heapq.heappush(pq, (-root_ub, 0, counter, 0, 0, 0, ()))

    best_val = 0
    best_solutions = []
    nodes_generated, nodes_expanded, nodes_pruned = 1, 0, 0

    while pq:
        neg_ub, _, _, idx, val, weight, selected = heapq.heappop(pq)
        ub = -neg_ub

        # Pruning condition
        if find_all_optima:
            if ub < best_val - 1e-9:
                nodes_pruned += 1
                continue
        else:
            if ub <= best_val:
                nodes_pruned += 1
                continue

        if idx == len(projects):
            continue

        nodes_expanded += 1
        name, proj_val, proj_weight = projects[idx]

        # Branch 1: Include Project (Left Child)
        if weight + proj_weight <= budget:
            in_val = val + proj_val
            in_w = weight + proj_weight
            in_sel = selected + (name,)
            in_ub = get_fractional_upper_bound(idx + 1, in_val, in_w, projects, budget)
            counter += 1
            nodes_generated += 1

            if in_val > best_val:
                best_val = in_val
                best_solutions = [(in_val, in_w, in_sel)]
            elif in_val == best_val and find_all_optima:
                best_solutions.append((in_val, in_w, in_sel))

            should_prune = (in_ub < best_val - 1e-9) if find_all_optima else (in_ub <= best_val)
            if should_prune:
                nodes_pruned += 1
            else:
                heapq.heappush(pq, (-in_ub, -in_val, counter, idx + 1, in_val, in_w, in_sel))

        # Branch 2: Exclude Project (Right Child)
        ex_val = val
        ex_w = weight
        ex_sel = selected
        ex_ub = get_fractional_upper_bound(idx + 1, ex_val, ex_w, projects, budget)
        counter += 1
        nodes_generated += 1

        should_prune = (ex_ub < best_val - 1e-9) if find_all_optima else (ex_ub <= best_val)
        if should_prune:
            nodes_pruned += 1
        else:
            heapq.heappush(pq, (-ex_ub, -ex_val, counter, idx + 1, ex_val, ex_w, ex_sel))

    return best_val, best_solutions, nodes_generated, nodes_expanded, nodes_pruned


if __name__ == "__main__":
    print("--- DAA Lab 12: Resource Budget Planning (Branch and Bound) ---\n")

    # 1. Solve sorted by descending benefit-to-resource ratio (Best Practice)
    sorted_by_ratio = sorted(PROJECTS, key=lambda p: p[1] / p[2], reverse=True)
    best_v, solutions, gen, exp, prun = solve_resource_planning(sorted_by_ratio, find_all_optima=True)

    print(f"Total Available Budget : {TOTAL_BUDGET} units")
    print(f"Maximum Total Benefit  : {best_v}")
    print(f"\nOptimal Project Subsets Found ({len(solutions)} alternate optima):")
    for idx, (val, wt, projs) in enumerate(solutions, 1):
        print(f"  Option {idx}: Projects {projs} | Used Resource = {wt}/{TOTAL_BUDGET} | Benefit = {val}")

    # 2. Impact of Node Handling (UB == best_val)
    print("\n" + "=" * 75)
    print("INVESTIGATION 1: Node Handling when UB == best_known_benefit")
    print("=" * 75)
    _, s_single, g_s, e_s, p_s = solve_resource_planning(sorted_by_ratio, find_all_optima=False)
    _, s_all, g_a, e_a, p_a = solve_resource_planning(sorted_by_ratio, find_all_optima=True)
    print(f"Prune if UB <= best (Single Optimum): Exp={e_s}, Gen={g_s}, Pruned={p_s} -> Found {len(s_single)} solution: {s_single[0][2]}")
    print(f"Prune if UB <  best (All Optima)   : Exp={e_a}, Gen={g_a}, Pruned={p_a} -> Found {len(s_all)} solutions: {[s[2] for s in s_all]}")

    # 3. Investigation: Project Ordering Impact on Search Nodes
    print("\n" + "=" * 75)
    print("INVESTIGATION 2: Impact of Project Consideration Ordering")
    print("=" * 75)
    orderings = [
        ("Descending Ratio (v/w)", sorted(PROJECTS, key=lambda p: p[1] / p[2], reverse=True)),
        ("Original Given Order", PROJECTS),
        ("Descending Benefit (v)", sorted(PROJECTS, key=lambda p: p[1], reverse=True)),
        ("Ascending Resource (w)", sorted(PROJECTS, key=lambda p: p[2])),
        ("Ascending Ratio (Worst)", sorted(PROJECTS, key=lambda p: p[1] / p[2]))
    ]

    print(f"{'Ordering Strategy':<25} | {'Generated':<10} | {'Expanded':<10} | {'Pruned':<10} | {'Max Benefit':<12}")
    print("-" * 75)
    for name, ord_p in orderings:
        v, _, g, e, p = solve_resource_planning(ord_p, find_all_optima=False)
        print(f"{name:<25} | {g:<10} | {e:<10} | {p:<10} | {v:<12}")
    print("=" * 75 + "\n")
