"""
Assignment 9:
Consider a long, quiet country road with houses scattered sparsely along it. Picture the road as
a line segment with an eastern and a western endpoint. Despite the setting, all residents are avid
cell phone users. You want to place cell phone base stations (towers) at certain points along
the road so that every house is within 4 miles of one of the base stations. Give an efficient
algorithm that achieves this goal using as few base stations as possible and implement your
algorithm.
"""

from typing import Dict, List, Tuple


def place_base_stations(
    house_positions: List[float],
    coverage_radius: float = 4.0,
) -> Tuple[int, List[float], Dict[float, List[float]]]:
    """
    Finds the minimum number and positions of cell phone base stations such that
    every house is within coverage_radius (4 miles) of at least one station.

    Parameters:
        house_positions (List[float]): Coordinates of houses along the road in miles.
        coverage_radius (float): Radius of coverage per base station (default: 4.0 miles).

    Returns:
        Tuple[int, List[float], Dict[float, List[float]]]:
            - min_stations (int): Minimum number of base stations required.
            - station_locations (List[float]): Selected locations for the base stations.
            - coverage_map (Dict[float, List[float]]): Mapping from each station to houses covered.

    Greedy Strategy:
        1. Sort house positions: h_1 <= h_2 <= ... <= h_k.
        2. Let h_i be the leftmost uncovered house.
           To cover h_i and maximize rightward coverage, place the tower at tower_pos = h_i + R.
        3. This tower covers all houses in [tower_pos - R, tower_pos + R] = [h_i, h_i + 2*R].
        4. Advance past all houses covered by this tower.
        5. Repeat for the next uncovered house.

    Complexity:
        - Sorting: O(k log k) where k is the number of houses.
        - Greedy Scan: O(k) single pass.
        - Total Time Complexity: O(k log k).
        - Auxiliary Space: O(k).
    """
    if not house_positions:
        return 0, [], {}

    sorted_houses = sorted(house_positions)
    k = len(sorted_houses)

    station_locations: List[float] = []
    coverage_map: Dict[float, List[float]] = {}

    idx = 0
    while idx < k:
        uncovered_house = sorted_houses[idx]

        # Place base station as far east as possible while covering uncovered_house
        tower_pos = uncovered_house + coverage_radius
        station_locations.append(tower_pos)
        coverage_map[tower_pos] = []

        # Cover all houses within [tower_pos - radius, tower_pos + radius]
        max_reach = tower_pos + coverage_radius
        while idx < k and sorted_houses[idx] <= max_reach:
            coverage_map[tower_pos].append(sorted_houses[idx])
            idx += 1

    return len(station_locations), station_locations, coverage_map


if __name__ == "__main__":
    print("=" * 80)
    print("Assignment 9: Cell Phone Base Station Placement on Country Road")
    print("=" * 80)

    # Sample road: Houses scattered along a 40-mile country road
    sample_houses = [1.5, 3.0, 4.5, 8.0, 10.5, 14.0, 18.2, 23.0, 27.5, 30.0, 31.5, 36.0]
    radius = 4.0

    print(f"Houses along road (miles) : {sample_houses}")
    print(f"Coverage Radius           : {radius} miles (covers within +/- {radius} miles)\n")

    num_towers, tower_locs, cov_map = place_base_stations(sample_houses, coverage_radius=radius)

    print(f"Minimum Base Stations Needed: {num_towers}")
    print("-" * 80)
    for i, loc in enumerate(tower_locs, 1):
        houses_covered = cov_map[loc]
        cov_range = f"[{loc - radius:04.1f} - {loc + radius:04.1f}] miles"
        print(f"Tower #{i:<2} at mile {loc:4.1f} | Range: {cov_range} | Covers {len(houses_covered)} houses: {houses_covered}")

    print("-" * 80)

    # Verification: Ensure every house is within radius miles of at least one tower
    for h in sample_houses:
        min_dist_to_tower = min(abs(h - t) for t in tower_locs)
        assert min_dist_to_tower <= radius + 1e-9, f"House at {h} is not covered! Min dist: {min_dist_to_tower}"

    print("Verification Passed: Every house is strictly within 4.0 miles of an assigned base station!")
    print("=" * 80)
