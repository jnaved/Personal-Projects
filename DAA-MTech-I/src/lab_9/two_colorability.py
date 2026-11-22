"""
Assignment 4:
A graph is k-colorable if each vertex can be given one of k colours and no edge connects
identically coloured vertices. Give a linear-time algorithm to test a graph for two-colorability.
Assume graphs are stored in adjacency-list format and specify any additional data structures
that are needed. How is a two-colorable graph related to a bipartite graph? Provide at least
two applications of two-colorability or bipartite graphs.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class Graph:
    """Undirected graph stored in adjacency-list format."""

    def __init__(self):
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self.vertices: Set[int] = set()

    def add_edge(self, u: int, v: int) -> None:
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.vertices.add(u)
        self.vertices.add(v)


def test_two_colorability(
    graph: Graph,
) -> Tuple[bool, Optional[Dict[int, int]], Optional[Tuple[List[int], List[int]]], str]:
    """
    Tests if an undirected graph stored in adjacency-list format is 2-colorable.

    Additional Data Structures:
        1. colors (Dict[int, int]): Maps each vertex to 0, 1, or -1 (uncolored).
        2. queue (deque): FIFO queue for breadth-first search (BFS) level traversal.

    Algorithm (Linear-Time BFS):
        1. Initialize all vertices as uncolored (-1).
        2. Iterate over all vertices in the graph to handle disconnected components.
        3. For any unvisited vertex, assign color 0 and start BFS:
           - Dequeue vertex u.
           - For each neighbor v:
             * If v is uncolored: assign color[v] = 1 - color[u], enqueue v.
             * If v is already colored and color[v] == color[u]:
               Conflict found! An odd-length cycle exists -> Not 2-colorable.
        4. If BFS completes with no conflicts, the graph is 2-colorable.

    Time Complexity:
        O(V + E) since every vertex is enqueued once and every edge is explored once.

    Space Complexity:
        O(V) for the colors mapping and the BFS queue.
    """
    colors: Dict[int, int] = {v: -1 for v in graph.vertices}

    for start_node in graph.vertices:
        if colors[start_node] != -1:
            continue

        # Start BFS for this connected component
        colors[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()

            for neighbor in graph.adj[u]:
                if colors[neighbor] == -1:
                    colors[neighbor] = 1 - colors[u]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[u]:
                    return (
                        False,
                        None,
                        None,
                        f"Odd-length cycle detected: edge ({u}, {neighbor}) connects vertices of identical color {colors[u]}.",
                    )

    # Partition vertices into two independent sets L and R
    set_L = [v for v, c in colors.items() if c == 0]
    set_R = [v for v, c in colors.items() if c == 1]

    return True, colors, (set_L, set_R), "Graph is 2-colorable (bipartite)."


def print_theoretical_notes():
    """Prints relationship to bipartite graphs and practical applications."""
    print("\n--- Theoretical Questions & Answers ---")
    print("1. Relationship to Bipartite Graphs:")
    print("   A graph is 2-colorable if and only if it is bipartite.")
    print("   Every bipartite graph can be partitioned into two disjoint independent sets L and R")
    print("   such that every edge connects a vertex in L to a vertex in R. Assigning Color 0 to L")
    print("   and Color 1 to R yields a valid 2-coloring. Both properties are mathematically equivalent")
    print("   to the graph containing NO odd-length cycles.")
    print("\n2. Two Major Applications of 2-Colorability / Bipartite Graphs:")
    print("   Application 1: Resource Matching & Assignment Problems")
    print("     - Maximum bipartite matching (e.g. allocating medical residents to hospitals, job applicants")
    print("       to open positions, or tasks to cloud computing servers).")
    print("   Application 2: Conflict-Free Scheduling & Frequency Allocation")
    print("     - Determining if exam sessions, wireless transmission channels, or railway tracks can be")
    print("       scheduled in two non-conflicting time slots or dual frequencies without collision.")


if __name__ == "__main__":
    print("=" * 75)
    print("Assignment 4: Linear-Time 2-Colorability Test (Bipartiteness)")
    print("=" * 75)

    # Graph 1: 2-Colorable (Bipartite Graph / Even Cycle C4)
    g1 = Graph()
    for u, v in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        g1.add_edge(u, v)

    is_2col1, colors1, partition1, msg1 = test_two_colorability(g1)
    print(f"Graph 1 (4-Cycle): {msg1}")
    print(f" - Color Assignments: {colors1}")
    print(f" - Bipartite Sets   : Set L = {partition1[0]}, Set R = {partition1[1]}")
    assert is_2col1 is True
    print("Verification Passed: Correctly identified 2-colorable graph!\n")

    # Graph 2: Non 2-Colorable (Odd Cycle C3 / Triangle)
    g2 = Graph()
    for u, v in [(1, 2), (2, 3), (3, 1)]:
        g2.add_edge(u, v)

    is_2col2, _, _, msg2 = test_two_colorability(g2)
    print(f"Graph 2 (Triangle / Odd Cycle): {msg2}")
    assert is_2col2 is False
    print("Verification Passed: Correctly rejected odd-cycle graph!")

    # Print theoretical questions answers
    print_theoretical_notes()
    print("=" * 75)
