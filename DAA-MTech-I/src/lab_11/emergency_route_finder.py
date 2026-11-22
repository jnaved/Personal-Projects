"""
========================================================================================
DAA Lab 11 - Assignment 2: Emergency Route Finder Using Dijkstra's Algorithm
========================================================================================

Problem Statement:
------------------
An ambulance service needs to find the shortest travel-time route from its current
location to a hospital. The city is represented as a weighted graph, where:
- Intersections and key locations (ambulance station, hospital) are vertices.
- Roads connecting intersections are edges weighted by travel time (e.g., in minutes).

Implement Dijkstra's algorithm to determine:
1. The minimum total travel time to the destination hospital.
2. The exact sequence of intersections (the shortest path) for the emergency vehicle.
3. Turn-by-turn navigation with cumulative travel times.

Algorithm:
----------
Dijkstra's Algorithm (Single-Source Shortest Paths with Non-Negative Weights):
1. Initialize distances from start location to all vertices as infinity, and distance to
   the start vertex as 0.
2. Maintain a min-priority queue (min-heap) of pairs (dist, vertex).
3. At each step, pop the vertex u with the smallest known distance.
4. If this distance is greater than the recorded shortest distance, skip it (outdated entry).
5. For each neighbor v along an outgoing road with travel time w:
   If dist[u] + w < dist[v]:
       dist[v] = dist[u] + w
       parent[v] = u
       push (dist[v], v) to priority queue.
6. Reconstruct the optimal path by backtracking parent pointers from destination to start.
========================================================================================
"""

import heapq
from typing import Dict, List, Optional, Tuple


class CityMap:
    """Represents a city road network with intersections and road travel times."""

    def __init__(self):
        # Adjacency list: node -> list of (neighbor, travel_time)
        self.adj: Dict[str, List[Tuple[str, float]]] = {}

    def add_intersection(self, name: str) -> None:
        """Add an intersection/location if not already present."""
        if name not in self.adj:
            self.adj[name] = []

    def add_road(self, u: str, v: str, travel_time: float, bidirectional: bool = True) -> None:
        """
        Add a road between intersections u and v with a specified travel time.
        Non-negative travel time is required.
        """
        if travel_time < 0:
            raise ValueError("Travel time must be non-negative.")
        self.add_intersection(u)
        self.add_intersection(v)
        self.adj[u].append((v, travel_time))
        if bidirectional:
            self.adj[v].append((u, travel_time))

    def dijkstra(
        self, start: str, destination: Optional[str] = None
    ) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """
        Runs Dijkstra's algorithm from the given start location.
        Returns:
            distances: mapping from each node to its shortest travel time from start.
            predecessors: mapping from each node to its predecessor on the shortest path.
        """
        if start not in self.adj:
            raise ValueError(f"Start location '{start}' does not exist in the city map.")

        distances: Dict[str, float] = {node: float("inf") for node in self.adj}
        predecessors: Dict[str, Optional[str]] = {node: None for node in self.adj}
        distances[start] = 0.0

        # Min-heap stores tuples of (distance, node)
        pq: List[Tuple[float, str]] = [(0.0, start)]

        while pq:
            curr_dist, u = heapq.heappop(pq)

            if curr_dist > distances[u]:
                continue

            # Early exit if we only care about reaching destination
            if destination is not None and u == destination:
                break

            for v, weight in self.adj[u]:
                new_dist = curr_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))

        return distances, predecessors

    def find_shortest_emergency_route(
        self, start: str, destination: str
    ) -> Tuple[Optional[List[str]], float, List[Tuple[str, str, float, float]]]:
        """
        Finds the quickest emergency route between start and destination.
        Returns:
            path: list of intersections on the route, or None if unreachable.
            total_time: shortest travel time in minutes.
            step_details: list of (from_node, to_node, segment_time, cumulative_time).
        """
        distances, predecessors = self.dijkstra(start, destination)

        if distances.get(destination, float("inf")) == float("inf"):
            return None, float("inf"), []

        # Reconstruct path from destination to start
        path: List[str] = []
        curr: Optional[str] = destination
        while curr is not None:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()

        # Build turn-by-turn segment details
        step_details: List[Tuple[str, str, float, float]] = []
        for i in range(len(path) - 1):
            u_node = path[i]
            v_node = path[i + 1]
            seg_time = distances[v_node] - distances[u_node]
            step_details.append((u_node, v_node, seg_time, distances[v_node]))

        return path, distances[destination], step_details

    def find_nearest_hospital(
        self, start: str, hospital_list: List[str]
    ) -> Tuple[Optional[str], Optional[List[str]], float]:
        """
        Given multiple hospitals, finds the one that can be reached fastest.
        """
        distances, predecessors = self.dijkstra(start)

        nearest_hosp = None
        min_time = float("inf")

        for hosp in hospital_list:
            if hosp in distances and distances[hosp] < min_time:
                min_time = distances[hosp]
                nearest_hosp = hosp

        if nearest_hosp is None or min_time == float("inf"):
            return None, None, float("inf")

        path: List[str] = []
        curr: Optional[str] = nearest_hosp
        while curr is not None:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()

        return nearest_hosp, path, min_time


# ========================================================================================
# Demonstration / Driver Code
# ========================================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ASSIGNMENT 2: EMERGENCY ROUTE FINDER USING DIJKSTRA'S ALGORITHM")
    print("=" * 80)

    city = CityMap()

    # Define city road network with travel times (in minutes)
    city.add_road("Ambulance_Station", "North_Cross", travel_time=4.0)
    city.add_road("Ambulance_Station", "Central_Square", travel_time=6.0)
    city.add_road("North_Cross", "East_Gate", travel_time=3.0)
    city.add_road("North_Cross", "Central_Square", travel_time=1.5)
    city.add_road("Central_Square", "South_Junction", travel_time=5.0)
    city.add_road("Central_Square", "West_Bridge", travel_time=7.0)
    city.add_road("East_Gate", "Apollo_Hospital", travel_time=6.0)
    city.add_road("East_Gate", "South_Junction", travel_time=2.0)
    city.add_road("South_Junction", "City_General_Hospital", travel_time=3.5)
    city.add_road("South_Junction", "Apollo_Hospital", travel_time=4.0)
    city.add_road("West_Bridge", "City_General_Hospital", travel_time=4.0)

    start_loc = "Ambulance_Station"
    target_hospital = "City_General_Hospital"

    print(f"\nDispatch Request:")
    print(f"  Current Location : {start_loc}")
    print(f"  Destination      : {target_hospital}")

    path, total_time, segments = city.find_shortest_emergency_route(start_loc, target_hospital)

    if path is None:
        print(f"\n[ALERT] No accessible route found from {start_loc} to {target_hospital}!")
    else:
        print("\n" + "=" * 60)
        print("OPTIMAL EMERGENCY NAVIGATION PLAN:")
        print("=" * 60)
        print(f"Total Travel Time: {total_time:.2f} minutes")
        print(f"Route Path       : {' -> '.join(path)}\n")

        print("Turn-by-Turn Directions:")
        print("-" * 60)
        print(f"{'Segment':<5} | {'From':<18} -> {'To':<22} | {'Time':<7} | {'Elapsed':<7}")
        print("-" * 60)
        for idx, (u, v, seg_time, cum_time) in enumerate(segments, start=1):
            print(f"Step {idx:<2} | {u:<18} -> {v:<22} | {seg_time:>4.1f} m  | {cum_time:>4.1f} m")
        print("-" * 60)

    # Secondary feature: Finding the quickest hospital among multiple options
    hospitals = ["Apollo_Hospital", "City_General_Hospital"]
    nearest_hosp, nearest_path, nearest_time = city.find_nearest_hospital(start_loc, hospitals)
    print("\n" + "=" * 60)
    print("NEAREST EMERGENCY HOSPITAL SELECTION:")
    print("=" * 60)
    print(f"Available Hospitals : {', '.join(hospitals)}")
    print(f"Quickest Reachable  : {nearest_hosp}")
    print(f"Minimum Travel Time : {nearest_time:.2f} minutes")
    print(f"Recommended Route   : {' -> '.join(nearest_path) if nearest_path else 'None'}")
    print("=" * 80)
