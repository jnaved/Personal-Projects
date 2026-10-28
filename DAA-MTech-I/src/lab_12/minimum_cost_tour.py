"""
========================================================================================
DAA Lab 12: Traveling Salesperson Problem (TSP) / Minimum-Cost Tour using Branch & Bound
========================================================================================

Problem Statement:
------------------
A delivery vehicle must start from City 1, visit every other city (Cities 2, 3, 4, 5)
exactly once, and return to City 1.
The travel costs between cities are given by the symmetric cost matrix C:

             City 1   City 2   City 3   City 4   City 5
City 1 (0) [   0       14        4       10       20   ]
City 2 (1) [  14        0        7        8        7   ]
City 3 (2) [   4        7        0        7       16   ]
City 4 (3) [  10        8        7        0        2   ]
City 5 (4) [  20        7       16        2        0   ]

A cost of 0 indicates that a city cannot connect to itself.

Find the minimum-cost Hamiltonian cycle (tour) using the Branch and Bound algorithm.

----------------------------------------------------------------------------------------
1. Design of the Lower-Bound Function (Degree-2 Relaxation for Symmetric TSP):
----------------------------------------------------------------------------------------
In any valid tour on an undirected graph, every city must have degree 2 (exactly two
incident edges in the tour: one arrival edge and one departure edge).

Sum of Two Smallest Incident Edges Principle:
For each vertex v, let e1(v) and e2(v) be the two cheapest edges incident to v.
The sum of edges chosen for v in any tour is >= e1(v) + e2(v).
Summing over all n vertices counts every tour edge twice (once for each endpoint).
Therefore, a valid lower bound on any tour is:
    Root LB = ceil( 0.5 * sum_{v in V} (e1(v) + e2(v)) )

For Root Node (Path = [City 1]):
  - City 1: two smallest edges are 4, 10  (sum = 14)
  - City 2: two smallest edges are 7, 7   (sum = 14)
  - City 3: two smallest edges are 4, 7   (sum = 11)
  - City 4: two smallest edges are 2, 7   (sum = 9)
  - City 5: two smallest edges are 2, 7   (sum = 9)
  Sum = 14 + 14 + 11 + 9 + 9 = 57
  Root LB = ceil(57 / 2) = 29.

For a Partial Tour Path P = (v_0, v_1, ..., v_k):
  - Path Cost Fixed: cost_fixed = sum of edges in path P.
  - Interior vertices (v_1 to v_{k-1}) have BOTH incident edges fixed.
  - Start endpoint v_0 has 1 edge fixed; needs 1 smallest edge to an unvisited city.
  - End endpoint v_k has 1 edge fixed; needs 1 smallest edge to an unvisited city.
  - Unvisited vertices u not in P have 0 edges fixed; need 2 smallest edges among
    unvisited cities and path endpoints {v_0, v_k}.
  
Lower Bound Formula:
  LB(P) = ceil( 0.5 * ( 2 * cost_fixed + edge_sum_unfixed ) )

Why is this admissible?
Every valid tour completion must complete the degree-2 requirement for every vertex
using valid edges. Since we select the minimum available edge costs for each vertex,
LB(P) <= True Cost of any completion of path P.

----------------------------------------------------------------------------------------
2. How Current Best Complete Tour Affects Subsequent Pruning Decisions:
----------------------------------------------------------------------------------------
- The algorithm maintains 'best_cost' (the cost of the cheapest complete tour found
  so far, initialized to infinity).
- As soon as the first complete tour is discovered:
      Tour: City 1 -> City 3 -> City 2 -> City 5 -> City 4 -> City 1
      Cost: 4 + 7 + 7 + 2 + 10 = 30
  'best_cost' drops from infinity to 30.
- Subsequent Pruning Criterion:
  Any active node with LB >= best_cost (i.e. LB >= 30) cannot strictly improve upon 30.
  Therefore, it is immediately PRUNED (discarded).
- In our search, branches like:
    * Path (1, 2) with LB = 34 >= 30 -> PRUNED
    * Path (1, 5) with LB = 40 >= 30 -> PRUNED
    * Path (1, 3, 5) with LB = 39 >= 30 -> PRUNED
  are eliminated without ever exploring their children!

----------------------------------------------------------------------------------------
3. Justification Why the Final Tour is Optimal:
----------------------------------------------------------------------------------------
1. Admissibility Guarantee:
   The lower-bound function never overestimates the cost of any tour completion
   (LB <= actual tour cost). Thus, no optimal solution was ever pruned.
2. Exhaustive Boundary Verification:
   When LCBB terminates, every live node in the search tree has either been fully
   explored or safely pruned with LB >= 30.
3. No Unexplored Path Can Cost Less:
   Since all non-explored paths have a mathematical lower bound >= 30, no tour in the
   entire combinatorial space can have a cost < 30.
4. Hence, the tour [1, 3, 2, 5, 4, 1] with cost 30 is globally optimal.
========================================================================================
"""

import heapq
from collections import deque

# Symmetric Cost Matrix C (0-indexed: Cities 0, 1, 2, 3, 4 correspond to Cities 1, 2, 3, 4, 5)
C = [
    [0, 14, 4, 10, 20],
    [14, 0, 7, 8, 7],
    [4, 7, 0, 7, 16],
    [10, 8, 7, 0, 2],
    [20, 7, 16, 2, 0]
]
N = len(C)


def calc_lower_bound(path):
    """
    Computes the degree-2 relaxation lower bound for a partial tour path.
    """
    k = len(path)
    # If all cities visited, return actual complete tour cost (return to start city 0)
    if k == N:
        return sum(C[path[i]][path[i + 1]] for i in range(N - 1)) + C[path[-1]][path[0]]

    # Fixed path cost
    cost_fixed = sum(C[path[i]][path[i + 1]] for i in range(k - 1))
    unvisited = set(range(N)) - set(path)
    sum_unfixed = 0

    if k == 1:
        # Root node: each vertex needs 2 smallest incident edges
        for v in range(N):
            edges = sorted([C[v][u] for u in range(N) if u != v])
            sum_unfixed += edges[0] + edges[1]
    else:
        # Start endpoint path[0] needs 1 edge to an unvisited city
        sum_unfixed += min(C[path[0]][u] for u in unvisited)
        # End endpoint path[-1] needs 1 edge to an unvisited city
        sum_unfixed += min(C[path[-1]][u] for u in unvisited)

        # Each unvisited city needs 2 edges among unvisited cities and path endpoints
        valid_neighbors = unvisited | {path[0], path[-1]}
        for v in unvisited:
            edges = sorted([C[v][u] for u in valid_neighbors if u != v])
            sum_unfixed += edges[0] + edges[1]

    # Ceil of half the total edge sum
    return (2 * cost_fixed + sum_unfixed + 1) // 2


def solve_tsp_lcbb():
    """
    Least-Cost Branch and Bound (Min-Heap / Priority Queue).
    Always expands the node with the least lower bound.
    """
    # Heap stores: (lb, counter, path)
    pq = []
    counter = 0
    root_lb = calc_lower_bound((0,))
    heapq.heappush(pq, (root_lb, counter, (0,)))

    best_cost = float('inf')
    best_tour = None
    nodes_generated, nodes_expanded, nodes_pruned = 1, 0, 0

    while pq:
        lb, _, path = heapq.heappop(pq)

        # Prune if lower bound meets or exceeds the best known complete tour cost
        if lb >= best_cost:
            nodes_pruned += 1
            continue

        # Complete tour found
        if len(path) == N:
            tour_cost = sum(C[path[i]][path[i + 1]] for i in range(N - 1)) + C[path[-1]][path[0]]
            if tour_cost < best_cost:
                best_cost = tour_cost
                best_tour = path + (path[0],)
            continue

        nodes_expanded += 1
        unvisited = [v for v in range(N) if v not in path]

        for next_city in unvisited:
            counter += 1
            nodes_generated += 1
            child_path = path + (next_city,)
            child_lb = calc_lower_bound(child_path)

            if child_lb >= best_cost:
                nodes_pruned += 1
            else:
                heapq.heappush(pq, (child_lb, counter, child_path))

    return best_cost, best_tour, nodes_generated, nodes_expanded, nodes_pruned


def solve_tsp_fifo():
    """
    FIFO Branch and Bound (Queue / Breadth-First Search).
    Expands nodes in the order they are generated.
    """
    queue = deque([(calc_lower_bound((0,)), (0,))])
    best_cost = float('inf')
    best_tour = None
    nodes_generated, nodes_expanded, nodes_pruned = 1, 0, 0

    while queue:
        lb, path = queue.popleft()

        if lb >= best_cost:
            nodes_pruned += 1
            continue

        if len(path) == N:
            tour_cost = sum(C[path[i]][path[i + 1]] for i in range(N - 1)) + C[path[-1]][path[0]]
            if tour_cost < best_cost:
                best_cost = tour_cost
                best_tour = path + (path[0],)
            continue

        nodes_expanded += 1
        unvisited = [v for v in range(N) if v not in path]

        for next_city in unvisited:
            nodes_generated += 1
            child_path = path + (next_city,)
            child_lb = calc_lower_bound(child_path)

            if child_lb >= best_cost:
                nodes_pruned += 1
            else:
                queue.append((child_lb, child_path))

    return best_cost, best_tour, nodes_generated, nodes_expanded, nodes_pruned


if __name__ == "__main__":
    print("--- DAA Lab 12: Minimum-Cost Tour (TSP) using Branch and Bound ---\n")

    # 1. Run LCBB (Least Lower Bound)
    lcbb_cost, lcbb_tour, lcbb_gen, lcbb_exp, lcbb_prune = solve_tsp_lcbb()

    # 2. Run FIFO (Generation Order)
    fifo_cost, fifo_tour, fifo_gen, fifo_exp, fifo_prune = solve_tsp_fifo()

    # Display 1-indexed tour for assignment output
    tour_1indexed = [city + 1 for city in lcbb_tour]
    print(f"Optimal Tour (1-indexed) : {' -> '.join(map(str, tour_1indexed))}")
    print(f"Minimum Tour Cost        : {lcbb_cost}")

    print("\nTour Edge Breakdown:")
    for i in range(len(lcbb_tour) - 1):
        u, v = lcbb_tour[i], lcbb_tour[i + 1]
        print(f"  City {u + 1} -> City {v + 1} : cost = {C[u][v]}")
    print(f"  Total Cost = {lcbb_cost}")

    print("\nSearch Efficiency Comparison:")
    print(f"{'Metric':<20} | {'FIFO (Order of Gen)':<20} | {'LCBB (Least LB)':<18}")
    print("-" * 64)
    print(f"{'Nodes Generated':<20} | {fifo_gen:<20} | {lcbb_gen:<18}")
    print(f"{'Nodes Expanded':<20} | {fifo_exp:<20} | {lcbb_exp:<18}")
    print(f"{'Nodes Pruned':<20} | {fifo_prune:<20} | {lcbb_prune:<18}")
    print("-" * 64)
    print(f"LCBB reduced node expansions by {((fifo_exp - lcbb_exp) / fifo_exp) * 100:.1f}%\n")
