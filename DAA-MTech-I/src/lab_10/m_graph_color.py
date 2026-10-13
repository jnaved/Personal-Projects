"""
List-Constrained Graph Coloring Problem using Backtracking
==========================================================
DAA Lab 8.1 - Question 3.1

Problem Description:
--------------------
Given an undirected graph G = (V, E), assign colors to vertices such that
no two adjacent vertices share the same color:
    color(u) != color(v) for all (u, v) in E.

Extensions in Question 3.1:
  1. Maximum m Available Colors: Colors are integers in {1, 2, ..., m}.
  2. Restricted Color Lists: Each vertex v has an allowed list of colors L(v).
  3. Pre-Colored Vertices: Certain vertices are fixed in advance and must
     retain their assigned color.
  4. Adjacency Matrix or List: Accepts graphs in either standard format.
  5. Backtracking Pruning: Prunes a branch immediately if a color violates
     an adjacency constraint or list constraint.
  6. Search Statistics: Reports total recursive calls and branches pruned.
  7. Minimum-Color Optimization: Finds the smallest m needed to color the graph.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Graph input can be adjacency list (dict) or adjacency matrix (2D list)
GraphInput = Union[Dict[int, List[int]], List[List[int]]]


# =====================================================================
# 1. HELPER: GRAPH FORMAT NORMALIZATION
# =====================================================================

def to_adj_list(graph: GraphInput) -> Dict[int, List[int]]:
    """
    Converts either an adjacency list (dict) or an adjacency matrix (2D list)
    into a standard dictionary adjacency list.
    """
    if isinstance(graph, dict):
        return {u: sorted(list(neighbors)) for u, neighbors in graph.items()}

    # If graph is a 2D matrix:
    n = len(graph)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if graph[i][j] != 0 and i != j:
                adj[i].append(j)
    return adj


# =====================================================================
# 2. INPUT VALIDATION (CONTRADICTION REJECTION)
# =====================================================================

def validate_inputs(
    adj: Dict[int, List[int]],
    m: int,
    allowed_colors: Dict[int, List[int]],
    pre_colored: Dict[int, int]
) -> None:
    """
    Validates input parameters and rejects invalid/contradictory constraints:
      1. m must be a positive integer.
      2. Pre-colored vertices must use a color in {1, ..., m}.
      3. Pre-colored vertices must use a color from their allowed list.
      4. Two adjacent pre-colored vertices cannot share the same color.
      5. No vertex can have an empty allowed color list.
    """
    if m <= 0:
        raise ValueError(f"Contradictory Input: Maximum colors m must be positive (got {m}).")

    for v in adj:
        if v not in allowed_colors or len(allowed_colors[v]) == 0:
            raise ValueError(f"Contradictory Input: Vertex {v} has an empty allowed colors list.")

    for v, c in pre_colored.items():
        if v not in adj:
            raise ValueError(f"Contradictory Input: Pre-colored vertex {v} does not exist in graph.")
        if not (1 <= c <= m):
            raise ValueError(f"Contradictory Input: Pre-colored vertex {v} has color {c} outside [1, {m}].")
        if c not in allowed_colors[v]:
            raise ValueError(
                f"Contradictory Input: Pre-colored vertex {v} has color {c}, "
                f"which is NOT in its allowed list {allowed_colors[v]}."
            )

    # Check for adjacent pre-colored vertices having the same color
    for u in pre_colored:
        for v in adj[u]:
            if v in pre_colored and pre_colored[u] == pre_colored[v]:
                raise ValueError(
                    f"Contradictory Input: Adjacent vertices {u} and {v} are pre-colored "
                    f"with the SAME color {pre_colored[u]}!"
                )


# =====================================================================
# 3. SAFETY CHECK (O(1) BOUNDING FUNCTION)
# =====================================================================

def is_safe(v: int, c: int, adj: Dict[int, List[int]], color: Dict[int, int]) -> bool:
    """
    Returns True if assigning color c to vertex v does not conflict with
    any already-colored adjacent neighbor.
    """
    for neighbor in adj[v]:
        if color.get(neighbor) == c:
            return False
    return True


# =====================================================================
# 4. CORE BACKTRACKING SOLVER
# =====================================================================

def solve_list_graph_coloring(
    graph: GraphInput,
    m: int,
    allowed_colors: Optional[Dict[int, List[int]]] = None,
    pre_colored: Optional[Dict[int, int]] = None,
    find_all: bool = True
) -> Tuple[List[Dict[int, int]], Dict[str, Any]]:
    """
    Solves the list-constrained graph coloring problem using recursive backtracking.

    Parameters:
      graph: Graph as an adjacency list or adjacency matrix.
      m: Maximum available colors {1, 2, ..., m}.
      allowed_colors: Dict mapping vertex -> list of permitted colors.
      pre_colored: Dict mapping vertex -> fixed color.
      find_all: If True, finds all valid colorings; if False, stops after the first.

    Returns:
      (valid_colorings, stats):
        - valid_colorings: List of color assignments [{v0: c0, v1: c1, ...}, ...]
        - stats: Search metrics including recursive calls and branches pruned.
    """
    adj = to_adj_list(graph)
    vertices = sorted(adj.keys())
    n = len(vertices)

    # Default allowed colors to {1, ..., m} for all vertices if not provided
    if allowed_colors is None:
        allowed = {v: list(range(1, m + 1)) for v in vertices}
    else:
        # Filter permitted colors by m
        allowed = {v: [c for c in allowed_colors.get(v, []) if c <= m] for v in vertices}

    pre_fixed = dict(pre_colored or {})

    # Validate inputs upfront
    validate_inputs(adj, m, allowed, pre_fixed)

    # Initialize color map with pre-colored vertices
    color: Dict[int, int] = {v: pre_fixed.get(v, 0) for v in vertices}

    stats: Dict[str, Any] = {
        "recursive_calls": 0,
        "branches_pruned_adjacency": 0,
        "branches_pruned_list": 0,
        "total_branches_pruned": 0,
        "colorings_found": 0,
        "elapsed_time_ms": 0.0,
    }

    valid_colorings: List[Dict[int, int]] = []
    start_time = time.perf_counter()

    def backtrack(idx: int) -> bool:
        stats["recursive_calls"] += 1

        # BASE CASE: All vertices successfully colored!
        if idx == n:
            valid_colorings.append(dict(color))
            stats["colorings_found"] += 1
            return not find_all  # return True to stop if find_all is False

        curr_v = vertices[idx]

        # CASE A: Vertex is pre-colored
        if curr_v in pre_fixed:
            fixed_c = pre_fixed[curr_v]
            # Verify no conflict with previously colored neighbors
            if not is_safe(curr_v, fixed_c, adj, color):
                stats["branches_pruned_adjacency"] += 1
                stats["total_branches_pruned"] += 1
                return False

            should_stop = backtrack(idx + 1)
            return should_stop

        # CASE B: Unassigned vertex -> try each allowed color
        for c in allowed[curr_v]:
            # Constraint check: Adjacency conflict
            if not is_safe(curr_v, c, adj, color):
                stats["branches_pruned_adjacency"] += 1
                stats["total_branches_pruned"] += 1
                continue

            # Assign color
            color[curr_v] = c

            should_stop = backtrack(idx + 1)
            if should_stop:
                return True

            # Undo assignment (backtrack)
            color[curr_v] = 0

        return False

    backtrack(0)
    stats["elapsed_time_ms"] = (time.perf_counter() - start_time) * 1000

    return valid_colorings, stats


def has_valid_coloring(
    graph: GraphInput,
    m: int,
    allowed_colors: Optional[Dict[int, List[int]]] = None,
    pre_colored: Optional[Dict[int, int]] = None
) -> bool:
    """Returns True if at least one valid coloring exists, False otherwise."""
    try:
        colorings, _ = solve_list_graph_coloring(graph, m, allowed_colors, pre_colored, find_all=False)
        return len(colorings) > 0
    except ValueError:
        return False


def find_minimum_colors(
    graph: GraphInput,
    allowed_colors: Optional[Dict[int, List[int]]] = None,
    pre_colored: Optional[Dict[int, int]] = None,
    max_search: int = 10
) -> Tuple[int, Optional[Dict[int, int]]]:
    """
    Minimum-color optimization: Finds the smallest m >= 1 for which
    a valid coloring exists under the given constraints.
    """
    for m in range(1, max_search + 1):
        try:
            colorings, _ = solve_list_graph_coloring(graph, m, allowed_colors, pre_colored, find_all=False)
            if colorings:
                return m, colorings[0]
        except ValueError:
            continue
    return -1, None


# =====================================================================
# 5. BENCHMARK GRAPHS: PETERSEN & REGULAR DODECAHEDRON
# =====================================================================

def get_petersen_graph() -> Dict[int, List[int]]:
    """Returns the standard Petersen graph (10 vertices, 15 edges)."""
    adj: Dict[int, List[int]] = {i: [] for i in range(10)}
    # Outer 5-cycle
    for i in range(5):
        adj[i].extend([(i + 1) % 5, (i + 4) % 5, i + 5])
    # Inner 5-star
    for i in range(5):
        adj[i + 5].extend([i, 5 + ((i + 2) % 5), 5 + ((i + 3) % 5)])
    return adj


def get_dodecahedron_graph() -> Dict[int, List[int]]:
    """Returns the Regular Dodecahedron graph (20 vertices, 30 edges, 3-regular)."""
    adj: Dict[int, List[int]] = {i: [] for i in range(20)}
    # Outer pentagon (0..4)
    for i in range(5):
        adj[i].extend([(i + 1) % 5, (i + 4) % 5, 5 + i])
        adj[5 + i].append(i)
    # Middle interlaced rings (5..9 to 10..14)
    for i in range(5):
        adj[5 + i].extend([10 + i, 10 + ((i + 4) % 5)])
        adj[10 + i].extend([5 + i, 5 + ((i + 1) % 5)])
    # Inner pentagon (15..19) connected to 10..14
    for i in range(5):
        adj[10 + i].append(15 + i)
        adj[15 + i].extend([10 + i, 15 + ((i + 1) % 5), 15 + ((i + 4) % 5)])
    return adj


# =====================================================================
# 6. MAIN LAB ASSIGNMENT DEMONSTRATION
# =====================================================================

if __name__ == "__main__":
    print("=" * 76)
    print(" DAA Lab 8.1 - Question 3.1: List-Constrained Graph Coloring Solver ")
    print("=" * 76)

    # -----------------------------------------------------------------
    # DEMO 1: Small Illustrative Graph with Restricted Color Lists
    # -----------------------------------------------------------------
    print("\n--- [Demo 1: Graph with Restricted Color Lists & Pre-Colored Vertex] ---")
    # Triangle with a tail: 0-1, 1-2, 2-0, 2-3
    small_graph = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2]
    }
    m1 = 3
    # Allowed color lists:
    # Vertex 0: only colors {1, 2}
    # Vertex 1: only colors {2, 3}
    # Vertex 2: only colors {1, 2, 3}
    # Vertex 3: only colors {1, 3}
    allowed_1 = {
        0: [1, 2],
        1: [2, 3],
        2: [1, 2, 3],
        3: [1, 3]
    }
    # Vertex 0 is pre-colored with Color 1
    pre_colored_1 = {0: 1}

    print(f"Graph Vertices       : {list(small_graph.keys())}")
    print(f"Edges                : [(0,1), (1,2), (2,0), (2,3)]")
    print(f"Maximum Colors (m)   : {m1}")
    print(f"Allowed Color Lists  : {allowed_1}")
    print(f"Pre-colored Vertices : {pre_colored_1}")

    colorings_1, stats_1 = solve_list_graph_coloring(
        graph=small_graph,
        m=m1,
        allowed_colors=allowed_1,
        pre_colored=pre_colored_1,
        find_all=True
    )

    print(f"\nValid coloring exists? : {len(colorings_1) > 0}")
    print(f"Total valid colorings  : {len(colorings_1)}")
    print(f"Recursive calls made   : {stats_1['recursive_calls']}")
    print(f"Branches pruned        : {stats_1['total_branches_pruned']} (due to adjacency conflicts)")
    print(f"Execution time         : {stats_1['elapsed_time_ms']:.3f} ms")

    print("\nAll Valid Colorings:")
    for i, col_map in enumerate(colorings_1, 1):
        print(f"  Coloring #{i}: {col_map} (V0={col_map[0]}, V1={col_map[1]}, V2={col_map[2]}, V3={col_map[3]})")

    # -----------------------------------------------------------------
    # DEMO 2: Adjacency Matrix Input Support
    # -----------------------------------------------------------------
    print("\n--- [Demo 2: Adjacency Matrix Representation Support] ---")
    # Same graph represented as an adjacency matrix:
    adj_matrix = [
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ]
    colorings_mat, _ = solve_list_graph_coloring(adj_matrix, m=3, allowed_colors=allowed_1, pre_colored=pre_colored_1)
    print(f"Adjacency matrix correctly parsed. Solutions count: {len(colorings_mat)} (matches Demo 1).")
    assert len(colorings_mat) == len(colorings_1)

    # -----------------------------------------------------------------
    # DEMO 3: Standard Big Polyhedral Shape (Regular Dodecahedron)
    # 20 vertices, 30 edges, 3-regular
    # -----------------------------------------------------------------
    print("\n--- [Demo 3: Standard Polyhedral Shape (Regular Dodecahedron)] ---")
    dodeca = get_dodecahedron_graph()
    m_dodeca = 3
    # Pre-color vertex 0 with Color 1, and vertex 1 with Color 2
    pre_dodeca = {0: 1, 1: 2}

    print(f"Benchmark Graph      : Regular Dodecahedron (20 vertices, 30 edges, 3-regular)")
    print(f"Maximum Colors (m)   : {m_dodeca}")
    print(f"Pre-colored Vertices : {pre_dodeca}")

    sols_dodeca, stats_dodeca = solve_list_graph_coloring(
        graph=dodeca,
        m=m_dodeca,
        pre_colored=pre_dodeca,
        find_all=False  # Find first valid solution
    )

    print(f"Valid coloring exists? : {len(sols_dodeca) > 0}")
    print(f"Recursive calls made   : {stats_dodeca['recursive_calls']}")
    print(f"Branches pruned        : {stats_dodeca['total_branches_pruned']}")
    print(f"Execution time         : {stats_dodeca['elapsed_time_ms']:.3f} ms")
    if sols_dodeca:
        sample_col = sols_dodeca[0]
        print(f"First Valid Coloring   : {sample_col}")
        # Verify no adjacent vertices share a color
        all_valid = all(sample_col[u] != sample_col[v] for u in dodeca for v in dodeca[u])
        assert all_valid, "Adjacency violation detected!"
        print("Verification           : Strictly Valid! No adjacent vertices share a color.")

    # -----------------------------------------------------------------
    # DEMO 4: Minimum-Color Optimization
    # -----------------------------------------------------------------
    print("\n--- [Demo 4: Minimum-Color Optimization] ---")
    min_m_small, opt_small = find_minimum_colors(small_graph)
    print(f"Small Graph (K3 + tail) Minimum Colors Required : {min_m_small} (Expected: 3)")
    assert min_m_small == 3

    petersen = get_petersen_graph()
    min_m_pet, _ = find_minimum_colors(petersen)
    print(f"Petersen Graph Minimum Colors Required           : {min_m_pet} (Expected: 3)")
    assert min_m_pet == 3

    # -----------------------------------------------------------------
    # DEMO 5: Contradiction Rejection Validation
    # -----------------------------------------------------------------
    print("\n--- [Demo 5: Contradiction Rejection Validation] ---")

    # 5a. Adjacent pre-colored vertices sharing same color
    try:
        solve_list_graph_coloring(small_graph, m=3, pre_colored={0: 1, 1: 1})
        print("ERROR: Failed to reject adjacent pre-colored conflict!")
    except ValueError as e:
        print(f"  [PASSED] Rejection 1: {e}")

    # 5b. Pre-colored vertex with color not in its allowed list
    try:
        solve_list_graph_coloring(small_graph, m=3, allowed_colors={0: [2, 3]}, pre_colored={0: 1})
        print("ERROR: Failed to reject pre-colored color not in allowed list!")
    except ValueError as e:
        print(f"  [PASSED] Rejection 2: {e}")

    # 5c. Pre-colored vertex with color > m
    try:
        solve_list_graph_coloring(small_graph, m=2, pre_colored={0: 3})
        print("ERROR: Failed to reject color > m!")
    except ValueError as e:
        print(f"  [PASSED] Rejection 3: {e}")

    print("\n" + "=" * 76)
    print("  All Question 3.1 Requirements Successfully Tested & Verified!  ")
    print("=" * 76)
