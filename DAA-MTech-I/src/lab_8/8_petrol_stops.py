"""
Assignment 8:
The Director of IIT Jammu drives from Jammu to New Delhi along the Jammu-Delhi highway.
His car’s petrol tank, when full, holds enough gas to travel n kilometres, and his map gives the
distances between petrol pump stations on his route. The Director wishes to make as few petrol
station stops as possible along the way. Give an efficient method by which he can determine at
which petrol stations to stop and implement it.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PetrolStation:
    name: str
    distance_from_start: float


def find_minimum_petrol_stops(
    tank_capacity_km: float,
    stations: List[PetrolStation],
    destination_km: float,
    start_km: float = 0.0,
) -> Tuple[Optional[int], List[PetrolStation], str]:
    """
    Finds the minimum number of petrol station stops required to travel from start to destination.

    Parameters:
        tank_capacity_km (float): Maximum distance n the car can travel on a full tank.
        stations (List[PetrolStation]): List of petrol stations along the route with their cumulative distances.
        destination_km (float): Distance of New Delhi from the starting point (Jammu).
        start_km (float): Starting coordinate (default: 0.0 km).

    Returns:
        Tuple[Optional[int], List[PetrolStation], str]:
            - stop_count: Minimum number of stops (or None if destination is unreachable).
            - chosen_stops: List of PetrolStation objects where the Director should refuel.
            - message: Diagnostic status message.

    Greedy Strategy (Furthest Reachable Station):
        1. Start at Jammu (d = 0) with a full tank.
        2. At the current location, find the furthest station along the route that is reachable
           without running out of fuel (distance <= n).
        3. If the destination itself is within range n, drive directly to destination without stopping.
        4. If the next consecutive station is farther than n, the destination is unreachable.
        5. Otherwise, stop at the furthest reachable station, refill the tank to full, and repeat.

    Optimality Justification:
        Refueling at the furthest reachable station maximizes the remaining reach along the highway
        for all subsequent legs. Any earlier stop leaves the car with a full tank at a point behind,
        which can only cover a subset of the route reachable from the furthest station.
        Hence, this greedy choice property strictly minimizes the total number of stops.

    Time Complexity:
        O(m) where m is the number of petrol stations (assuming stations are sorted by distance;
        O(m log m) if unsorted).
    Space Complexity:
        O(m) to store the route and selected stops.
    """
    if tank_capacity_km <= 0:
        return None, [], "Error: Tank capacity must be strictly positive."

    # Sort stations by cumulative distance
    sorted_stations = sorted(stations, key=lambda s: s.distance_from_start)

    # Filter stations to those between start and destination
    valid_stations = [
        s for s in sorted_stations if start_km < s.distance_from_start < destination_km
    ]

    # Create full itinerary checkpoints: [Start, Station 1, ..., Station m, Destination]
    route_points = [(start_km, "Jammu (Start)")]
    for s in valid_stations:
        route_points.append((s.distance_from_start, s.name))
    route_points.append((destination_km, "New Delhi (Destination)"))

    num_points = len(route_points)
    current_idx = 0
    chosen_stations: List[PetrolStation] = []

    while current_idx < num_points - 1:
        # Check if destination is reachable from current position
        if route_points[-1][0] - route_points[current_idx][0] <= tank_capacity_km:
            break

        # Find the furthest reachable station
        next_idx = current_idx
        while (
            next_idx + 1 < num_points
            and route_points[next_idx + 1][0] - route_points[current_idx][0] <= tank_capacity_km
        ):
            next_idx += 1

        # If we cannot even reach the immediate next station
        if next_idx == current_idx:
            stuck_loc = route_points[current_idx][1]
            unreachable_target = route_points[current_idx + 1][1]
            dist_needed = route_points[current_idx + 1][0] - route_points[current_idx][0]
            return (
                None,
                [],
                f"Unreachable: Gap of {dist_needed:.1f} km between '{stuck_loc}' and '{unreachable_target}' exceeds tank capacity of {tank_capacity_km:.1f} km.",
            )

        # Stop and refuel at the furthest reachable station
        chosen_stations.append(valid_stations[next_idx - 1])
        current_idx = next_idx

    return len(chosen_stations), chosen_stations, "Success: Optimal refueling schedule generated."


if __name__ == "__main__":
    print("=" * 80)
    print("Assignment 8: Minimizing Petrol Station Stops on Jammu-Delhi Highway")
    print("=" * 80)

    # Highway from Jammu to New Delhi (Total ~ 585 km)
    jammu_delhi_distance = 585.0

    # Petrol stations along the route with distances from Jammu
    jammu_delhi_stations = [
        PetrolStation("Kathua Station", 85.0),
        PetrolStation("Pathankot Station", 112.0),
        PetrolStation("Dasuya Station", 160.0),
        PetrolStation("Jalandhar Station", 215.0),
        PetrolStation("Phagwara Station", 238.0),
        PetrolStation("Ludhiana Station", 275.0),
        PetrolStation("Khanna Station", 320.0),
        PetrolStation("Ambala Station", 395.0),
        PetrolStation("Kurukshetra Station", 440.0),
        PetrolStation("Karnal Station", 470.0),
        PetrolStation("Panipat Station", 505.0),
        PetrolStation("Sonipat Station", 545.0),
    ]

    print(f"Journey Route : Jammu (0 km) -> New Delhi ({jammu_delhi_distance} km)")
    print(f"Available Stations ({len(jammu_delhi_stations)} stations):")
    for s in jammu_delhi_stations:
        print(f"   - {s.name:<22}: {s.distance_from_start:5.1f} km from Jammu")

    # Scenario 1: Car with full tank range of n = 150 km
    n1 = 150.0
    print("\n" + "-" * 80)
    print(f"SCENARIO 1: Tank Range n = {n1} km")
    stops_count, stops_list, msg = find_minimum_petrol_stops(n1, jammu_delhi_stations, jammu_delhi_distance)
    print(f"Status               : {msg}")
    print(f"Minimum Stops Needed : {stops_count}")
    print("Refueling Stops Sequence:")
    prev_d = 0.0
    for i, s in enumerate(stops_list, 1):
        leg_dist = s.distance_from_start - prev_d
        print(f"   Stop #{i}: {s.name:<20} at km {s.distance_from_start:5.1f} (Drove {leg_dist:5.1f} km on this tank)")
        prev_d = s.distance_from_start
    final_leg = jammu_delhi_distance - prev_d
    print(f"   Final Leg to New Delhi: Drove {final_leg:5.1f} km to finish.")

    # Scenario 2: Car with larger tank range of n = 250 km
    n2 = 250.0
    print("\n" + "-" * 80)
    print(f"SCENARIO 2: Tank Range n = {n2} km")
    stops_count2, stops_list2, msg2 = find_minimum_petrol_stops(n2, jammu_delhi_stations, jammu_delhi_distance)
    print(f"Status               : {msg2}")
    print(f"Minimum Stops Needed : {stops_count2}")
    for i, s in enumerate(stops_list2, 1):
        print(f"   Stop #{i}: {s.name:<20} at km {s.distance_from_start:5.1f}")

    # Scenario 3: Impossible tank size (e.g. n = 40 km, unable to cross 85 km to Kathua)
    n3 = 40.0
    print("\n" + "-" * 80)
    print(f"SCENARIO 3: Insufficient Tank Range n = {n3} km (Testing Feasibility Check)")
    stops_count3, _, msg3 = find_minimum_petrol_stops(n3, jammu_delhi_stations, jammu_delhi_distance)
    print(f"Result : {msg3}")
    assert stops_count3 is None
    print("Verification Passed: Correctly reported unreachable route when gap > tank capacity.")
    print("=" * 80)
