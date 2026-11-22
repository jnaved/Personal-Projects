"""
========================================================================================
DAA Lab 11 - Assignment 1: Design a Minimum-Cost One-Way Delivery Network 
                            Using Minimum Arborescence
========================================================================================

Problem Statement:
------------------
A logistics company has one central warehouse and several regional distribution centers.
Possible one-way delivery routes have different costs. Design a minimum-cost directed
distribution structure in which:
1. Every regional center is reachable from the central warehouse (root).
2. Each non-root regional center has exactly one incoming route in the selected structure.
3. The total cost of the selected routes is minimized.

Algorithm:
----------
Chu-Liu / Edmonds' Algorithm for Minimum Spanning Arborescence:
1. For each non-root vertex v, select the incoming directed edge (u, v) with the minimum weight.
2. If any non-root vertex cannot be reached (has no incoming edge), no valid arborescence exists.
3. Check if the selected edges form any directed cycle.
   - If no cycle exists, these selected edges constitute the optimal minimum arborescence.
4. If a directed cycle C is detected:
   - Contract all vertices in C into a single super-vertex.
   - For every edge entering C from an outside vertex, adjust its weight by subtracting
     the weight of the cycle's internal incoming edge to that vertex.
   - Recursively solve the minimum arborescence problem on the contracted graph.
   - Expand the cycle: include all cycle edges except the one entering the vertex that
     received an external edge in the contracted solution.
5. Return the set of selected delivery routes and the total cost.
========================================================================================
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class DeliveryNetwork:
    """Represents a directed delivery network of warehouses and distribution centers."""

    def __init__(self, root: str):
        self.root: str = root
        self.nodes: Set[str] = {root}
        self.edges: List[Tuple[str, str, float]] = []

    def add_route(self, u: str, v: str, cost: float) -> None:
        """Add a directed route from distribution center u to center v with given cost."""
        self.nodes.add(u)
        self.nodes.add(v)
        self.edges.append((u, v, cost))

    def find_minimum_delivery_network(self) -> Tuple[Optional[List[Tuple[str, str, float]]], float]:
        """
        Computes the minimum-cost directed delivery network (minimum arborescence)
        rooted at the central warehouse using Chu-Liu / Edmonds' algorithm.
        """
        node_list = sorted(list(self.nodes))
        node_to_idx = {name: idx for idx, name in enumerate(node_list)}
        idx_to_name = {idx: name for idx, name in enumerate(node_list)}

        n = len(node_list)
        root_idx = node_to_idx[self.root]

        indexed_edges = [(node_to_idx[u], node_to_idx[v], cost) for u, v, cost in self.edges]

        chosen_edges_idx, total_cost = self._chu_liu_edmonds(n, indexed_edges, root_idx)

        if chosen_edges_idx is None:
            return None, float("inf")

        chosen_routes = [(idx_to_name[u], idx_to_name[v], cost) for u, v, cost in chosen_edges_idx]
        return chosen_routes, total_cost

    @staticmethod
    def _chu_liu_edmonds(
        n: int, orig_edges: List[Tuple[int, int, float]], root: int
    ) -> Tuple[Optional[List[Tuple[int, int, float]]], float]:
        """
        Core implementation of Chu-Liu / Edmonds' algorithm on integer-indexed vertices.
        """
        curr_n = n
        curr_root = root
        # Edge representation: (u, v, weight, original_edge_index)
        curr_edges = [(u, v, float(w), idx) for idx, (u, v, w) in enumerate(orig_edges)]

        history = []

        while True:
            # 1. Select cheapest incoming edge for each non-root vertex
            min_in: Dict[int, Tuple[int, int, float, int]] = {}
            for u, v, w, eid in curr_edges:
                if v == curr_root or u == v:
                    continue
                if v not in min_in or w < min_in[v][2]:
                    min_in[v] = (u, v, w, eid)

            # Check if all non-root nodes are reachable
            for v in range(curr_n):
                if v != curr_root and v not in min_in:
                    return None, float("inf")

            # 2. Check for directed cycles among chosen edges
            parent = {v: min_in[v][0] for v in min_in}
            cycle: List[int] = []
            visited = [0] * curr_n  # 0: unvisited, 1: visiting, 2: visited

            for i in range(curr_n):
                if i == curr_root or visited[i] != 0:
                    continue
                curr = i
                path = []
                while curr != curr_root and visited[curr] == 0:
                    visited[curr] = 1
                    path.append(curr)
                    curr = parent.get(curr, curr_root)

                if curr != curr_root and visited[curr] == 1:
                    idx = path.index(curr)
                    cycle = path[idx:]
                    break

                for node in path:
                    visited[node] = 2

            # If no cycle was formed, the chosen edges form a valid arborescence!
            if not cycle:
                chosen_eids = {min_in[v][3] for v in min_in}
                break

            # 3. Contract the cycle into a single super-node
            cycle_set = set(cycle)
            new_id = {}
            cnt = 0
            for v in range(curr_n):
                if v not in cycle_set:
                    new_id[v] = cnt
                    cnt += 1
            cycle_rep = cnt  # ID for contracted cycle
            for v in cycle:
                new_id[v] = cycle_rep

            # Track which node inside the cycle is targeted by external edges
            entering_node: Dict[int, int] = {}
            new_edges = []
            for u, v, w, eid in curr_edges:
                u_in = u in cycle_set
                v_in = v in cycle_set
                if u_in and v_in:
                    # Internal cycle edge (handled during cycle expansion)
                    continue
                elif not u_in and v_in:
                    # Edge entering the cycle from outside:
                    # New weight = original_weight - weight_of_incoming_cycle_edge
                    entering_node[eid] = v
                    new_w = w - min_in[v][2]
                    new_edges.append((new_id[u], cycle_rep, new_w, eid))
                else:
                    # Edge outside the cycle or leaving the cycle
                    new_edges.append((new_id[u], new_id[v], w, eid))

            history.append({
                "cycle": cycle,
                "min_in": min_in,
                "entering_node": entering_node,
            })

            curr_edges = new_edges
            curr_root = new_id[curr_root]
            curr_n = cnt + 1

        # 4. Unwind contracted cycles in reverse order
        for step in reversed(history):
            cycle = step["cycle"]
            min_in = step["min_in"]
            entering_node = step["entering_node"]

            # Identify which edge entering the cycle was selected
            v_star = None
            for eid in chosen_eids:
                if eid in entering_node:
                    v_star = entering_node[eid]
                    break

            # Include all internal cycle edges except the one into v_star
            for v in cycle:
                if v != v_star:
                    chosen_eids.add(min_in[v][3])

        chosen_edges = [orig_edges[eid] for eid in chosen_eids]
        total_cost = sum(w for _, _, w in chosen_edges)
        return chosen_edges, total_cost


def print_arborescence_structure(root: str, routes: List[Tuple[str, str, float]]) -> None:
    """Displays the selected routes as an indented hierarchy tree."""
    tree = defaultdict(list)
    for u, v, cost in routes:
        tree[u].append((v, cost))

    print(f"\nDelivery Tree Root: [{root}]")

    def _dfs(node: str, depth: int):
        for child, cost in sorted(tree[node]):
            prefix = "    " * depth + "|-- "
            print(f"{prefix}-> [{child}] (Route Cost: {cost})")
            _dfs(child, depth + 1)

    _dfs(root, 0)


# ========================================================================================
# Demonstration / Driver Code
# ========================================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ASSIGNMENT 1: MINIMUM-COST ONE-WAY DELIVERY NETWORK USING MINIMUM ARBORESCENCE")
    print("=" * 80)

    # Example: Central Warehouse distributing to 5 Regional Centers (A, B, C, D, E)
    # Graph contains directed cycles among centers, testing cycle contraction.
    network = DeliveryNetwork(root="Central_Warehouse")

    # Routes from Central Warehouse
    network.add_route("Central_Warehouse", "Center_A", cost=10.0)
    network.add_route("Central_Warehouse", "Center_B", cost=15.0)
    network.add_route("Central_Warehouse", "Center_C", cost=20.0)

    # Routes between regional centers
    network.add_route("Center_A", "Center_B", cost=2.0)
    network.add_route("Center_B", "Center_C", cost=3.0)
    network.add_route("Center_C", "Center_A", cost=4.0)  # Cycle A -> B -> C -> A
    network.add_route("Center_B", "Center_D", cost=5.0)
    network.add_route("Center_D", "Center_E", cost=6.0)
    network.add_route("Center_E", "Center_D", cost=7.0)  # Cycle D <-> E
    network.add_route("Center_C", "Center_E", cost=8.0)

    print("\nAvailable One-Way Delivery Routes:")
    print("-" * 50)
    for u, v, cost in network.edges:
        print(f"  {u:<18} -> {v:<12} | Cost: {cost}")

    routes, total_cost = network.find_minimum_delivery_network()

    if routes is None:
        print("\n[ERROR] Not all regional centers are reachable from the Central Warehouse.")
    else:
        print("\n" + "=" * 50)
        print("OPTIMAL DELIVERY NETWORK (Minimum Arborescence):")
        print("=" * 50)
        for u, v, cost in sorted(routes):
            print(f"  Route: {u:<18} -> {v:<12} (Cost: {cost})")

        print("-" * 50)
        print(f"Total Minimum Delivery Network Cost: {total_cost}")

        # Verify problem constraints:
        incoming_counts = defaultdict(int)
        for _, v, _ in routes:
            incoming_counts[v] += 1

        print("\nConstraint Verification:")
        print(f"  1. Central Warehouse incoming routes: {incoming_counts['Central_Warehouse']} (Expected: 0)")
        non_root_valid = all(incoming_counts[node] == 1 for node in network.nodes if node != "Central_Warehouse")
        print(f"  2. Each regional center has exactly 1 incoming route: {'PASS' if non_root_valid else 'FAIL'}")

        print_arborescence_structure("Central_Warehouse", routes)
        print("=" * 80)
