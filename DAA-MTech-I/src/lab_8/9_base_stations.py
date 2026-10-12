"""
Assignment 9:
Consider a long, quiet country road with houses scattered sparsely along it. Picture the road as
a line segment with an eastern and a western endpoint. Despite the setting, all residents are avid
cell phone users. You want to place cell phone base stations (towers) at certain points along
the road so that every house is within 4 miles of one of the base stations. Give an efficient
algorithm that achieves this goal using as few base stations as possible and implement your
algorithm.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
            - coverage_map (Dict[float, List[float]]): Mapping from each station location to the list of houses it covers.

    Greedy Strategy:
        1. Sort house positions from west to east: h_1 <= h_2 <= ... <= h_k.
        2. Let h_i be the leftmost uncovered house.
           To cover h_i, a tower must be placed within [h_i - R, h_i + R].
           To maximize coverage for houses to the east, place the tower as far east as possible:
           tower_pos = h_i + R.
        3. This tower covers all houses in [tower_pos - R, tower_pos + R] = [h_i, h_i + 2*R].
        4. Skip all houses covered by this tower.
        5. Repeat for the next uncovered house until all houses are covered.

    Proof of Correctness (Greedy Exchange Argument):
        - Any valid solution must cover the leftmost uncovered house h_i.
        - Therefore, the tower covering h_i must be located at some coordinate t <= h_i + R.
        - The greedy choice sets t* = h_i + R >= t.
        - The set of houses covered by t* to the right is [t* - R, t* + R] = [h_i, h_i + 2R],
          which is a superset of the houses covered to the right by t ([t - R, t + R] where t + R <= h_i + 2R).
        - No houses to the left of h_i need coverage because they are already covered.
        - Thus, substituting t with t* never hurts and leaves fewer or equal remaining houses to cover.
        - By induction, the greedy placement uses the minimum possible number of base stations.

    Complexity:
        - Sorting: O(k log k) where k is the number of houses.
        - Greedy Scan: O(k) single pass.
        - Total Time Complexity: O(k log k).
        - Auxiliary Space: O(k).
    """
    if not house_positions:
        return 0, [], {}

    # Sort house positions
    sorted_houses = sorted(house_positions)
    k = len(sorted_houses)

    station_locations: List[float] = []
    coverage_map: Dict[float, List[float]] = {}

    idx = 0
    while idx < k:
        # Leftmost uncovered house
        uncovered_house = sorted_houses[idx]

        # Place base station as far east as possible while still covering uncovered_house
        tower_pos = uncovered_house + coverage_radius
        station_locations.append(tower_pos)
        coverage_map[tower_pos] = []

        # Cover all houses within [tower_pos - radius, tower_pos + radius]
        # which corresponds to [uncovered_house, uncovered_house + 2 * coverage_radius]
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
        print(f"Tower #{i:<2} at mile {loc:4.1f} | Coverage Range: {cov_range} | Covers {len(houses_covered)} houses: {houses_covered}")

    print("-" * 80)

    # Verification: Ensure every house is within 4 miles of at least one tower
    for h in sample_houses:
        min_dist_to_tower = min(abs(h - t) for t in tower_locs)
        assert min_dist_to_tower <= radius + 1e-9, f"House at {h} is not covered! Min dist: {min_dist_to_tower}"

    print("Verification Passed: Every house is strictly within 4.0 miles of an assigned base station!")
    print("=" * 80)
