"""
Assignment 1:
Suppose a firm is to manage the construction of billboards on the Jammu-Srinagar Highway,
which runs South-North for M kilometres. The possible billboard sites are given by numbers
x1, x2, . . . , xn within the interval [0,M]. The position xi indicates the position of the billboards
along the highway, measured from its western end. If it is decided to place a billboard at
location xi, the firm receives a revenue of ri > 0. Regulations of the Jammu and Kashmir
governments require that no two billboards be placed within less than or equal to 5 kilometres
of each other. As part of the optimization problem, the firm must decide where to place the
billboards at a subset of the sites x1, x2, . . . , xn to maximize the total revenue, subject to this
constraint. Give an algorithm that takes in an instance of this problem and returns the maximum
total revenue that can be obtained from any valid subset of sites. Also, give its running time.

Example:
Suppose M = 20, n = 4, [x1, x2, x3, x4] = [6, 7, 12, 14], and [r1, r2, r3, r4] = [5, 6, 5, 1].
Then, the optimal solution is to place billboards at x1 and x3 for a total revenue of 10.
"""

import bisect
import sys
from pathlib import Path
from typing import List, Tuple

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def max_billboard_revenue(
    m: float,
    positions: List[float],
    revenues: List[float],
    min_dist: float = 5.0,
) -> Tuple[float, List[float]]:
    """
    Computes the maximum revenue and optimal placement of billboards along a highway of length M
    such that no two billboards are placed within <= min_dist kilometres of each other (|xi - xj| > min_dist).

    Parameters:
        m (float): Total length of the highway in kilometres.
        positions (List[float]): Positions of potential billboard sites in [0, M].
        revenues (List[float]): Revenue obtained from placing a billboard at each site.
        min_dist (float): Minimum separation distance required (strictly greater than 5 km).

    Returns:
        Tuple[float, List[float]]: (max_revenue, chosen_positions)

    Algorithm:
        1. Pair positions and revenues, sort by position: O(n log n).
        2. For each site i, find predecessor p(i): largest index j < i with positions[i] - positions[j] > min_dist.
           Using binary search: O(n log n) total.
        3. Recurrence:
           OPT(i) = max(OPT(i-1), revenue[i] + OPT(p(i)))
           where OPT(-1) = 0.
        4. Backtrack to reconstruct the optimal billboard sites: O(n).

    Time Complexity:
        - Sorting: O(n log n)
        - Predecessor search: O(n log n) via binary search (or O(n) using two pointers)
        - DP table filling: O(n)
        - Backtracking: O(n)
        Total Time Complexity: O(n log n) where n is the number of sites.
        (Independent of M, strongly polynomial).

    Space Complexity:
        O(n) for the DP array and predecessor array.
    """
    n = len(positions)
    if n == 0 or m <= 0:
        return 0.0, []

    # Filter sites that fall strictly within [0, M] and pair with revenue
    paired = []
    for x, r in zip(positions, revenues):
        if 0.0 <= x <= m and r > 0:
            paired.append((x, r))

    if not paired:
        return 0.0, []

    # Sort sites in increasing order of highway position
    paired.sort(key=lambda item: item[0])
    sorted_pos = [item[0] for item in paired]
    sorted_rev = [item[1] for item in paired]
    k = len(paired)

    # p[i]: index of the latest site before i that is > min_dist km away
    # Condition: sorted_pos[i] - sorted_pos[j] > min_dist <=> sorted_pos[j] < sorted_pos[i] - min_dist
    p = [-1] * k
    eps = 1e-9  # For numerical stability with floating point positions
    for i in range(k):
        target = sorted_pos[i] - min_dist - eps
        # bisect_right returns insertion point after any <= target
        idx = bisect.bisect_right(sorted_pos, target, 0, i) - 1
        p[i] = idx if idx >= 0 else -1

    # dp[i] = max revenue achievable from a valid subset of sites 0 ... i
    dp = [0.0] * k
    dp[0] = sorted_rev[0]

    for i in range(1, k):
        incl = sorted_rev[i] + (dp[p[i]] if p[i] != -1 else 0.0)
        excl = dp[i - 1]
        dp[i] = max(incl, excl)

    # Reconstruct optimal placement
    chosen_sites = []
    curr = k - 1
    while curr >= 0:
        incl = sorted_rev[curr] + (dp[p[curr]] if p[curr] != -1 else 0.0)
        excl = dp[curr - 1] if curr > 0 else 0.0

        if incl >= excl:
            chosen_sites.append(sorted_pos[curr])
            curr = p[curr]
        else:
            curr -= 1

    chosen_sites.reverse()
    return dp[k - 1], chosen_sites


if __name__ == "__main__":
    print("=" * 70)
    print("Assignment 1: Billboard Placement on Jammu-Srinagar Highway")
    print("=" * 70)

    # Test with the example given in the problem statement:
    # M = 20, n = 4, positions = [6, 7, 12, 14], revenues = [5, 6, 5, 1]
    M = 20
    positions = [6, 7, 12, 14]
    revenues = [5, 6, 5, 1]

    max_rev, selected = max_billboard_revenue(M, positions, revenues, min_dist=5.0)

    print(f"Highway Length (M) : {M} km")
    print(f"Potential Sites    : {positions}")
    print(f"Revenues           : {revenues}")
    print(f"Minimum Distance   : > 5.0 km")
    print("-" * 70)
    print(f"Optimal Placement  : {selected}")
    print(f"Maximum Revenue    : {max_rev}")
    print("-" * 70)

    # Verify example outcome
    assert max_rev == 10, f"Expected 10, got {max_rev}"
    assert selected == [6, 12], f"Expected [6, 12], got {selected}"
    print("Verification Passed: Correctly obtained revenue 10 with billboards at [6, 12].")

    # Additional Test: Densely spaced sites
    M2 = 30
    pos2 = [2, 5, 8, 11, 15, 21, 27]
    rev2 = [10, 8, 12, 7, 15, 20, 25]
    rev_test, sel_test = max_billboard_revenue(M2, pos2, rev2, min_dist=5.0)
    print(f"\nExtended Test:")
    print(f"Sites: {pos2}, Revenues: {rev2}")
    print(f"Selected: {sel_test}, Max Revenue: {rev_test}")
    for i in range(len(sel_test) - 1):
        assert sel_test[i+1] - sel_test[i] > 5.0, "Violation of distance constraint!"
    print("All separation constraints strictly satisfied!")
    print("=" * 70)
