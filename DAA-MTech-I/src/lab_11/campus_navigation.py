"""
========================================================================================
DAA Lab 11 - Assignment 5: Campus Navigation Using Breadth-First Search (BFS)
========================================================================================

Problem Statement:
------------------
A university wants a navigation system that finds the route with the fewest walkway
segments between campus buildings.
- Buildings are represented as vertices.
- Walkways connecting adjacent buildings are represented as unweighted edges.

Implement Breadth-First Search (BFS) to:
1. Find the minimum number of walkway segments from a starting building to a destination.
2. Reconstruct the complete step-by-step route between the two buildings.
3. Handle edge cases such as start == destination, or unreachable buildings.

Algorithm (Breadth-First Search for Unweighted Shortest Paths):
--------------------------------------------------------------
Since all walkway segments have equal unit weight (1 segment each), BFS is guaranteed
to find the path with the fewest edges in O(V + E) time.
1. Use a FIFO queue initialized with the starting building.
2. Maintain a `visited` set and a `parent` dictionary for path reconstruction.
3. Maintain a `distance` dictionary recording the number of walkway segments from start.
4. For each visited building u, inspect all its adjacent buildings v via walkways:
   - If v has not been visited:
     * Mark v as visited.
     * Record distance[v] = distance[u] + 1.
     * Record parent[v] = u.
     * Enqueue v.
     * If v is the destination, terminate early.
5. Reconstruct the route by backtracking from destination to start using parent pointers.
========================================================================================
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class CampusGraph:
    """Represents a university campus layout with buildings and walkways."""

    def __init__(self):
        # Adjacency list: building -> list of connected neighbor buildings
        self.adj: Dict[str, List[str]] = defaultdict(list)
        self.buildings: Set[str] = set()

    def add_building(self, name: str) -> None:
        """Register a building on campus."""
        self.buildings.add(name)
        # Ensure it appears in adjacency list even if isolated
        _ = self.adj[name]

    def add_walkway(self, b1: str, b2: str) -> None:
        """Add an unweighted bidirectional walkway connecting two buildings."""
        self.add_building(b1)
        self.add_building(b2)
        self.adj[b1].append(b2)
        self.adj[b2].append(b1)

    def find_shortest_route(
        self, start: str, destination: str
    ) -> Tuple[Optional[int], Optional[List[str]]]:
        """
        Uses Breadth-First Search (BFS) to find the route with the fewest walkway segments.
        Returns:
            (min_segments, path) where path is the ordered list of buildings visited.
            Returns (None, None) if destination is unreachable.
        """
        if start not in self.buildings:
            raise ValueError(f"Start building '{start}' does not exist on campus.")
        if destination not in self.buildings:
            raise ValueError(f"Destination building '{destination}' does not exist on campus.")

        # Trivial case: already at destination
        if start == destination:
            return 0, [start]

        visited: Set[str] = {start}
        parent: Dict[str, Optional[str]] = {start: None}
        distance: Dict[str, int] = {start: 0}
        queue: deque[str] = deque([start])

        found = False
        while queue:
            curr = queue.popleft()

            if curr == destination:
                found = True
                break

            for neighbor in self.adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    distance[neighbor] = distance[curr] + 1
                    queue.append(neighbor)
                    if neighbor == destination:
                        found = True
                        break
            if found:
                break

        if not found and destination not in distance:
            return None, None

        # Reconstruct path by following parent pointers
        path: List[str] = []
        curr_node: Optional[str] = destination
        while curr_node is not None:
            path.append(curr_node)
            curr_node = parent[curr_node]
        path.reverse()

        return distance[destination], path

    def find_buildings_within_distance(self, start: str, max_segments: int) -> Dict[str, int]:
        """Finds all campus buildings reachable within a specified number of walkway segments."""
        if start not in self.buildings:
            raise ValueError(f"Start building '{start}' not found.")

        visited = {start}
        distance = {start: 0}
        queue = deque([start])

        while queue:
            curr = queue.popleft()
            if distance[curr] >= max_segments:
                continue

            for neighbor in self.adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distance[neighbor] = distance[curr] + 1
                    queue.append(neighbor)

        return distance


# ========================================================================================
# Demonstration / Driver Code
# ========================================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ASSIGNMENT 5: CAMPUS NAVIGATION USING BREADTH-FIRST SEARCH (BFS)")
    print("=" * 80)

    campus = CampusGraph()

    # Define campus walkways (unweighted edges between buildings)
    walkways = [
        ("Main_Gate", "Admin_Building"),
        ("Main_Gate", "Security_Post"),
        ("Admin_Building", "Central_Library"),
        ("Admin_Building", "Auditorium"),
        ("Central_Library", "Computer_Science_Dept"),
        ("Central_Library", "Student_Center"),
        ("Computer_Science_Dept", "Electronics_Block"),
        ("Computer_Science_Dept", "Mechanical_Block"),
        ("Electronics_Block", "Robotics_Lab"),
        ("Mechanical_Block", "Workshop"),
        ("Student_Center", "Cafeteria"),
        ("Student_Center", "Sports_Complex"),
        ("Cafeteria", "Hostel_Block_A"),
        ("Hostel_Block_A", "Hostel_Block_B"),
        ("Hostel_Block_B", "Sports_Complex"),
        ("Sports_Complex", "Health_Center"),
    ]

    for b1, b2 in walkways:
        campus.add_walkway(b1, b2)

    # Isolated research facility on an off-campus parcel
    campus.add_building("Off_Campus_Observatory")

    print(f"\nCampus Map loaded with {len(campus.buildings)} buildings and {len(walkways)} walkways.")

    # -----------------------------------------------------------------------------------
    # Query 1: Hostel to Computer Science Department
    # -----------------------------------------------------------------------------------
    start1 = "Hostel_Block_A"
    dest1 = "Computer_Science_Dept"
    print("\n" + "-" * 75)
    print(f"Navigation Query 1: Route from '{start1}' to '{dest1}'")
    segments1, path1 = campus.find_shortest_route(start1, dest1)

    if path1 is not None and segments1 is not None:
        print(f"Fewest Walkway Segments : {segments1} segments")
        print(f"Optimal Navigation Path : {' -> '.join(path1)}")
        print("\nTurn-by-turn navigation steps:")
        for idx in range(len(path1) - 1):
            print(f"  Step {idx + 1}: Walk from [{path1[idx]}] to [{path1[idx + 1]}]")
    else:
        print(f"No walkway connection exists between {start1} and {dest1}.")

    # -----------------------------------------------------------------------------------
    # Query 2: Main Gate to Sports Complex
    # -----------------------------------------------------------------------------------
    start2 = "Main_Gate"
    dest2 = "Sports_Complex"
    print("\n" + "-" * 75)
    print(f"Navigation Query 2: Route from '{start2}' to '{dest2}'")
    segments2, path2 = campus.find_shortest_route(start2, dest2)

    if path2 is not None and segments2 is not None:
        print(f"Fewest Walkway Segments : {segments2} segments")
        print(f"Optimal Navigation Path : {' -> '.join(path2)}")
    else:
        print(f"No walkway connection exists between {start2} and {dest2}.")

    # -----------------------------------------------------------------------------------
    # Query 3: Start == Destination
    # -----------------------------------------------------------------------------------
    start3 = "Central_Library"
    dest3 = "Central_Library"
    print("\n" + "-" * 75)
    print(f"Navigation Query 3: Route from '{start3}' to '{dest3}'")
    segments3, path3 = campus.find_shortest_route(start3, dest3)
    print(f"Segments: {segments3}, Path: {' -> '.join(path3)}")

    # -----------------------------------------------------------------------------------
    # Query 4: Unreachable / Disconnected Building
    # -----------------------------------------------------------------------------------
    start4 = "Main_Gate"
    dest4 = "Off_Campus_Observatory"
    print("\n" + "-" * 75)
    print(f"Navigation Query 4: Route from '{start4}' to '{dest4}'")
    segments4, path4 = campus.find_shortest_route(start4, dest4)
    if path4 is None:
        print(f"Result: Destination [{dest4}] is unreachable on foot from [{start4}].")

    # -----------------------------------------------------------------------------------
    # Bonus Feature: Buildings reachable within 2 walkway segments from Student Center
    # -----------------------------------------------------------------------------------
    center_hub = "Student_Center"
    print("\n" + "-" * 75)
    print(f"Walkability Radius: Buildings within 2 walkway segments from '{center_hub}':")
    nearby = campus.find_buildings_within_distance(center_hub, max_segments=2)
    for bldg, seg_count in sorted(nearby.items(), key=lambda item: (item[1], item[0])):
        if bldg != center_hub:
            print(f"  * {bldg:<25} : {seg_count} segment(s) away")

    print("=" * 80)
