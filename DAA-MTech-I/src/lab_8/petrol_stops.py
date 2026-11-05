"""
Assignment 8:
The Director of IIT Jammu drives from Jammu to New Delhi along the Jammu-Delhi highway.
His car’s petrol tank, when full, holds enough gas to travel n kilometres, and his map gives the
distances between petrol pump stations on his route. The Director wishes to make as few petrol
station stops as possible along the way. Give an efficient method by which he can determine at
which petrol stations to stop and implement it.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


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
        stations (List[PetrolStation]): List of petrol stations along the route with cumulative distances.
        destination_km (float): Distance of New Delhi from the starting point (Jammu).
        start_km (float): Starting coordinate (default: 0.0 km).

    Returns:
        Tuple[Optional[int], List[PetrolStation], str]:
            - stop_count: Minimum number of stops (or None if destination is unreachable).
            - chosen_stops: List of PetrolStation objects where refueling occurs.
            - message: Status message.

    Greedy Strategy (Furthest Reachable Station):
        1. Start at Jammu (d = 0) with a full tank.
        2. At current location, find the furthest station reachable without running out of fuel (dist <= n).
        3. If destination is reachable directly, proceed without stopping.
        4. If next consecutive station is farther than n, destination is unreachable.
        5. Otherwise, stop at the furthest reachable station, refuel, and repeat.

    Complexity:
        Time Complexity: O(m) where m is the number of petrol stations (if sorted; O(m log m) if unsorted).
        Space Complexity: O(m) to store route and chosen stops.
    """
    if tank_capacity_km <= 0:
        return None, [], "Error: Tank capacity must be strictly positive."

    # Sort stations by cumulative distance
    sorted_stations = sorted(stations, key=lambda s: s.distance_from_start)

    # Filter stations between start and destination
    valid_stations = [
        s for s in sorted_stations if start_km < s.distance_from_start < destination_km
    ]

    # Full route: [Start, Station 1, ..., Station m, Destination]
    route_points = [(start_km, "Jammu (Start)")]
    for s in valid_stations:
        route_points.append((s.distance_from_start, s.name))
    route_points.append((destination_km, "New Delhi (Destination)"))

    num_points = len(route_points)
    current_idx = 0
    chosen_stations: List[PetrolStation] = []

    while current_idx < num_points - 1:
        # Destination reachable from current position
        if route_points[-1][0] - route_points[current_idx][0] <= tank_capacity_km:
            break

        # Find the furthest reachable station
        next_idx = current_idx
        while (
            next_idx + 1 < num_points
            and route_points[next_idx + 1][0] - route_points[current_idx][0] <= tank_capacity_km
        ):
            next_idx += 1

        # Cannot reach the next station
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

    # Highway from Jammu to New Delhi (~585 km)
    jammu_delhi_distance = 585.0

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
    print(f"Available Stations: {len(jammu_delhi_stations)} stations along route")

    # Full tank range: n = 150 km
    n = 150.0
    print(f"Tank Range    : n = {n} km")
    print("-" * 80)

    stops_count, stops_list, msg = find_minimum_petrol_stops(n, jammu_delhi_stations, jammu_delhi_distance)
    print(f"Status               : {msg}")
    print(f"Minimum Stops Needed : {stops_count}")
    print("Refueling Stops Sequence:")
    prev_d = 0.0
    for i, s in enumerate(stops_list, 1):
        leg_dist = s.distance_from_start - prev_d
        print(f"   Stop #{i}: {s.name:<20} at km {s.distance_from_start:5.1f} (Drove {leg_dist:5.1f} km on this tank)")
        prev_d = s.distance_from_start

    final_leg = jammu_delhi_distance - prev_d
    print(f"   Final Leg to Destination: Drove {final_leg:5.1f} km to New Delhi.")

    assert stops_count == 4, f"Expected 4 stops, got {stops_count}"
    print("-" * 80)
    print("Verification Passed: Optimal refueling schedule generated with 4 stops!")
    print("=" * 80)
