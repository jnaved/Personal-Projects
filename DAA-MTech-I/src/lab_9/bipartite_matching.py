"""
Assignment 5:
Edge-weighted graphs can be interpreted as a network of pipes, where the weight of edge (i, j)
determines the pipe’s capacity. The network flow problem asks for the maximum flow from a
source vertex s to a target vertex t in a weighted graph G, respecting the capacity of each pipe.
• Create a source node s connected to every vertex in set L by an edge of weight 1.
• Create a sink node t and connect it to every vertex in set R by an edge of weight 1.
• Assign each edge in the bipartite graph G a weight of 1.
Now, the maximum possible flow from s to t defines the largest matching in G. Implement the
algorithm.
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple


class FlowNetwork:
    """Residual flow network for computing maximum network flow via Edmonds-Karp."""

    def __init__(self):
        self.adj: Dict[str, List[str]] = defaultdict(list)
        self.capacity: Dict[Tuple[str, str], int] = defaultdict(int)
        self.flow: Dict[Tuple[str, str], int] = defaultdict(int)

    def add_edge(self, u: str, v: str, cap: int) -> None:
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.capacity[(u, v)] = cap
        self.capacity[(v, u)] = 0  # Residual backward capacity

    def bfs_find_path(self, source: str, sink: str, parent: Dict[str, str]) -> bool:
        """Finds an augmenting path from source to sink with available residual capacity."""
        visited: Set[str] = {source}
        queue = deque([source])

        while queue:
            curr = queue.popleft()
            if curr == sink:
                return True

            for neighbor in self.adj[curr]:
                residual = self.capacity[(curr, neighbor)] - self.flow[(curr, neighbor)]
                if neighbor not in visited and residual > 0:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        return False

    def edmonds_karp_max_flow(self, source: str, sink: str) -> int:
        """Computes maximum flow from source to sink using Edmonds-Karp algorithm."""
        max_flow = 0
        parent: Dict[str, str] = {}

        while self.bfs_find_path(source, sink, parent):
            # Find bottleneck capacity along augmenting path
            path_flow = float("inf")
            curr = sink
            while curr != source:
                p = parent[curr]
                residual = self.capacity[(p, curr)] - self.flow[(p, curr)]
                path_flow = min(path_flow, residual)
                curr = p

            # Update flow and reverse flow along the path
            curr = sink
            while curr != source:
                p = parent[curr]
                self.flow[(p, curr)] += path_flow
                self.flow[(curr, p)] -= path_flow
                curr = p

            max_flow += path_flow
            parent.clear()

        return max_flow


def max_bipartite_matching(
    left_set: List[str],
    right_set: List[str],
    bipartite_edges: List[Tuple[str, str]],
) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Computes Maximum Bipartite Matching by reducing to Max-Flow:
      1. Source node 's' connected to every vertex u in left_set with capacity 1.
      2. Sink node 't' connected from every vertex v in right_set with capacity 1.
      3. Every bipartite edge (u, v) assigned capacity 1 directed from left to right.
      4. Run Edmonds-Karp Max-Flow. The flow value equals maximum matching size.

    Complexity:
      Time Complexity: O(V * E) since all unit capacities limit max flow to at most |V|.
      Space Complexity: O(V + E).
    """
    network = FlowNetwork()
    source = "source_s"
    sink = "sink_t"

    # Connect source 's' to all vertices in L with capacity 1
    for u in left_set:
        network.add_edge(source, u, 1)

    # Add bipartite edges with capacity 1 from L to R
    for u, v in bipartite_edges:
        network.add_edge(u, v, 1)

    # Connect all vertices in R to sink 't' with capacity 1
    for v in right_set:
        network.add_edge(v, sink, 1)

    # Compute maximum flow
    max_flow = network.edmonds_karp_max_flow(source, sink)

    # Extract matching: edges between L and R with positive flow
    matched_edges: List[Tuple[str, str]] = []
    for u, v in bipartite_edges:
        if network.flow[(u, v)] == 1:
            matched_edges.append((u, v))

    return max_flow, matched_edges


if __name__ == "__main__":
    print("=" * 75)
    print("Assignment 5: Maximum Bipartite Matching via Max-Flow Reduction")
    print("=" * 75)

    # Example: Matching Applicants (L) to Job Roles (R)
    L = ["Alice", "Bob", "Charlie", "Diana"]
    R = ["Software Engineer", "Data Scientist", "Cybersecurity Analyst", "DevOps Engineer"]

    # Candidate qualifications (bipartite edges)
    edges = [
        ("Alice", "Software Engineer"),
        ("Alice", "Data Scientist"),
        ("Bob", "Software Engineer"),
        ("Bob", "Cybersecurity Analyst"),
        ("Charlie", "Data Scientist"),
        ("Diana", "Cybersecurity Analyst"),
        ("Diana", "DevOps Engineer"),
    ]

    print(f"Set L (Applicants) : {L}")
    print(f"Set R (Job Roles)  : {R}")
    print(f"Available Edges    : {len(edges)} candidate qualifications\n")

    max_matching_size, matching = max_bipartite_matching(L, R, edges)

    print(f"Maximum Matching Size (Max-Flow): {max_matching_size}")
    print("Optimal Assigned Pairs:")
    for applicant, role in matching:
        print(f"   * {applicant:<10} <---> {role}")

    # Verify matching validity (no applicant or role assigned twice)
    matched_l = [u for u, v in matching]
    matched_r = [v for u, v in matching]
    assert len(matched_l) == len(set(matched_l)), "Duplicate applicant in matching!"
    assert len(matched_r) == len(set(matched_r)), "Duplicate role in matching!"
    assert max_matching_size == 4, f"Expected 4, got {max_matching_size}"

    print("-" * 75)
    print("Verification Passed: Valid maximum cardinality bipartite matching obtained!")
    print("=" * 75)
