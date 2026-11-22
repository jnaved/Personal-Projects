"""
Assignment 6:
A person wants to fly from city A to city B. Suppose several airlines have connections in
different cities, and the flights operate on different schedules. The goal is to minimize travel
time or cost. Write a program that implements the shortest-path algorithm to help the person
make the most efficient travel arrangements.
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Flight:
    origin: str
    destination: str
    airline: str
    flight_num: str
    duration_hours: float
    cost_inr: float


class FlightNetwork:
    """Represents an airline flight network between cities."""

    def __init__(self):
        self.adj: Dict[str, List[Flight]] = defaultdict(list)
        self.cities: set[str] = set()

    def add_flight(
        self,
        origin: str,
        destination: str,
        airline: str,
        flight_num: str,
        duration_hours: float,
        cost_inr: float,
    ) -> None:
        flight = Flight(origin, destination, airline, flight_num, duration_hours, cost_inr)
        self.adj[origin].append(flight)
        self.cities.add(origin)
        self.cities.add(destination)

    def find_shortest_itinerary(
        self,
        source: str,
        target: str,
        criterion: str = "cost",
    ) -> Tuple[Optional[float], Optional[List[Flight]], str]:
        """
        Dijkstra's Algorithm to find the optimal travel itinerary from source to target city.

        Parameters:
            source (str): Starting city (City A).
            target (str): Destination city (City B).
            criterion (str): 'cost' to minimize monetary expense, or 'time' to minimize flight duration.

        Returns:
            Tuple[Optional[float], Optional[List[Flight]], str]:
                - total_metric: Minimal total cost or duration.
                - itinerary: List of connecting flights taken.
                - summary_msg: Status description.

        Complexity:
            Time Complexity: O((V + E) log V) via Min-Heap.
            Space Complexity: O(V + E).
        """
        if source not in self.cities or target not in self.cities:
            return None, None, f"One or both cities ({source}, {target}) not present in network."

        # Min-heap elements: (accumulated_metric, current_city)
        pq: List[Tuple[float, str]] = [(0.0, source)]
        dist: Dict[str, float] = {source: 0.0}
        prev_flight: Dict[str, Tuple[str, Flight]] = {}

        while pq:
            current_dist, curr_city = heapq.heappop(pq)

            if current_dist > dist.get(curr_city, float("inf")):
                continue

            if curr_city == target:
                break

            for flight in self.adj[curr_city]:
                weight = flight.cost_inr if criterion == "cost" else flight.duration_hours
                new_dist = current_dist + weight
                dest = flight.destination

                if new_dist < dist.get(dest, float("inf")):
                    dist[dest] = new_dist
                    prev_flight[dest] = (curr_city, flight)
                    heapq.heappush(pq, (new_dist, dest))

        if target not in dist:
            return None, None, f"No viable flight route connecting {source} to {target}."

        # Reconstruct path
        itinerary: List[Flight] = []
        curr = target
        while curr != source:
            prev_city, flight = prev_flight[curr]
            itinerary.append(flight)
            curr = prev_city

        itinerary.reverse()
        return dist[target], itinerary, f"Optimal route found optimizing for {criterion}."


if __name__ == "__main__":
    print("=" * 75)
    print("Assignment 6: Flight Route Optimization (Dijkstra's Algorithm)")
    print("=" * 75)

    network = FlightNetwork()

    # Flight network between Indian metropolitan and regional hubs
    flights_data = [
        ("Jammu", "Delhi", "Air India", "AI-822", 1.5, 4500.0),
        ("Jammu", "Srinagar", "IndiGo", "6E-212", 0.8, 2500.0),
        ("Srinagar", "Delhi", "SpiceJet", "SG-160", 1.4, 3800.0),
        ("Delhi", "Mumbai", "Vistara", "UK-995", 2.2, 5500.0),
        ("Delhi", "Bengaluru", "IndiGo", "6E-501", 2.8, 6200.0),
        ("Delhi", "Ahmedabad", "Air India", "AI-011", 1.5, 3200.0),
        ("Ahmedabad", "Mumbai", "Akasa Air", "QP-112", 1.2, 2100.0),
        ("Mumbai", "Bengaluru", "Air India", "AI-631", 1.8, 3800.0),
        ("Jammu", "Chandigarh", "Alliance Air", "9I-645", 1.1, 2800.0),
        ("Chandigarh", "Mumbai", "IndiGo", "6E-728", 2.4, 4900.0),
    ]

    for orig, dest, airline, fnum, dur, cost in flights_data:
        network.add_flight(orig, dest, airline, fnum, dur, cost)

    origin_city = "Jammu"
    destination_city = "Bengaluru"

    print(f"Journey Planning: {origin_city} ---> {destination_city}\n")

    # 1. Optimize for Cost
    print("--- 1. Optimization Criterion: MINIMUM COST (INR) ---")
    min_cost, cost_itin, msg_c = network.find_shortest_itinerary(origin_city, destination_city, criterion="cost")
    print(f"Status: {msg_c}")
    print(f"Total Cost: Rs. {min_cost:.2f}")
    total_time_on_cost_route = sum(f.duration_hours for f in cost_itin)
    print("Flight Route:")
    for i, f in enumerate(cost_itin, 1):
        print(f"  Leg {i}: {f.origin} -> {f.destination} via {f.airline} ({f.flight_num}) | Duration: {f.duration_hours}h | Cost: Rs. {f.cost_inr:.2f}")
    print(f"Total Flight Time: {total_time_on_cost_route:.1f} hours\n")

    # 2. Optimize for Flight Time
    print("--- 2. Optimization Criterion: MINIMUM TRAVEL TIME (Hours) ---")
    min_time, time_itin, msg_t = network.find_shortest_itinerary(origin_city, destination_city, criterion="time")
    print(f"Status: {msg_t}")
    print(f"Total Duration: {min_time:.1f} hours")
    total_cost_on_time_route = sum(f.cost_inr for f in time_itin)
    print("Flight Route:")
    for i, f in enumerate(time_itin, 1):
        print(f"  Leg {i}: {f.origin} -> {f.destination} via {f.airline} ({f.flight_num}) | Duration: {f.duration_hours}h | Cost: Rs. {f.cost_inr:.2f}")
    print(f"Total Monetary Cost: Rs. {total_cost_on_time_route:.2f}")

    assert cost_itin is not None and time_itin is not None
    print("-" * 75)
    print("Verification Passed: Dijkstra shortest-path routes successfully computed!")
    print("=" * 75)
