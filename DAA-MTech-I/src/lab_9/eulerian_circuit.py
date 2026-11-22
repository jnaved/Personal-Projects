"""
Assignment 2:
An Euler circuit in a directed graph is a cycle in which every edge is visited exactly once.
Design and implement a linear-time algorithm to find an Euler circuit in a directed graph
(where one exists).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class DirectedGraph:
    """Directed graph supporting Eulerian circuit detection and construction."""

    def __init__(self):
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self.in_degree: Dict[int, int] = defaultdict(int)
        self.out_degree: Dict[int, int] = defaultdict(int)
        self.num_edges = 0

    def add_edge(self, u: int, v: int) -> None:
        self.adj[u].append(v)
        self.out_degree[u] += 1
        self.in_degree[v] += 1
        # Ensure vertices exist in degrees dict
        _ = self.in_degree[u]
        _ = self.out_degree[v]
        self.num_edges += 1

    def is_eulerian(self) -> Tuple[bool, str, Optional[int]]:
        """
        A directed graph has an Eulerian circuit if and only if:
        1. In-degree equals out-degree for every vertex.
        2. All vertices with non-zero degree belong to a single connected component.
        """
        if self.num_edges == 0:
            return True, "Graph has no edges (trivially Eulerian).", None

        # 1. Degree balance check: in_degree(v) == out_degree(v) for all v
        all_vertices = set(self.in_degree.keys()) | set(self.out_degree.keys())
        start_vertex = None

        for v in all_vertices:
            if self.in_degree[v] != self.out_degree[v]:
                return False, f"Vertex {v} has in-degree {self.in_degree[v]} != out-degree {self.out_degree[v]}.", None
            if self.out_degree[v] > 0 and start_vertex is None:
                start_vertex = v

        if start_vertex is None:
            return True, "Trivially Eulerian.", None

        # 2. Connectivity check on non-zero degree vertices
        visited = set()

        def dfs(u: int):
            visited.add(u)
            for neighbor in self.adj[u]:
                if neighbor not in visited:
                    dfs(neighbor)

        dfs(start_vertex)

        for v in all_vertices:
            if self.out_degree[v] > 0 and v not in visited:
                return False, f"Graph is disconnected: vertex {v} unreachable from {start_vertex}.", None

        return True, "Eulerian circuit exists.", start_vertex

    def find_eulerian_circuit(self) -> Tuple[Optional[List[int]], str]:
        """
        Hierholzer's Algorithm:
        Constructs an Eulerian circuit in linear time O(V + E).
        """
        possible, msg, start_node = self.is_eulerian()
        if not possible:
            return None, msg

        if self.num_edges == 0 or start_node is None:
            return [], msg

        # Make a copy of adjacency lists so we can pop traversed edges in O(1)
        edge_lists = {u: list(neighbors) for u, neighbors in self.adj.items()}

        circuit: List[int] = []
        stack: List[int] = [start_node]

        while stack:
            curr = stack[-1]
            if edge_lists.get(curr):
                # Traverse an unused outgoing edge
                next_vertex = edge_lists[curr].pop()
                stack.append(next_vertex)
            else:
                # Backtrack and record in circuit
                circuit.append(stack.pop())

        circuit.reverse()

        # Circuit must traverse all edges (num_edges + 1 vertices in path)
        if len(circuit) != self.num_edges + 1:
            return None, "Not all edges were reachable in a single circuit."

        return circuit, "Eulerian circuit successfully constructed."


if __name__ == "__main__":
    print("=" * 75)
    print("Assignment 2: Eulerian Circuit in Directed Graph (Hierholzer's Algorithm)")
    print("=" * 75)

    # Graph 1: Valid Eulerian Directed Graph
    # Circuit: 0 -> 1 -> 2 -> 0 -> 3 -> 4 -> 0
    g1 = DirectedGraph()
    edges_g1 = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
    for u, v in edges_g1:
        g1.add_edge(u, v)

    circuit1, msg1 = g1.find_eulerian_circuit()
    print(f"Graph 1 Edges: {edges_g1}")
    print(f"Status       : {msg1}")
    print(f"Euler Circuit: {' -> '.join(map(str, circuit1))}")
    assert circuit1 is not None and len(circuit1) == len(edges_g1) + 1
    print("Verification Passed: All edges visited exactly once in cycle!\n")

    # Graph 2: Non-Eulerian Graph (unbalanced degrees)
    g2 = DirectedGraph()
    edges_g2 = [(0, 1), (1, 2), (2, 0), (2, 3)]
    for u, v in edges_g2:
        g2.add_edge(u, v)

    circuit2, msg2 = g2.find_eulerian_circuit()
    print(f"Graph 2 Edges: {edges_g2}")
    print(f"Status       : {msg2}")
    assert circuit2 is None
    print("Verification Passed: Correctly identified non-Eulerian directed graph.")
    print("=" * 75)
