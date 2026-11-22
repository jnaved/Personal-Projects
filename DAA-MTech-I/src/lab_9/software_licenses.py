"""
Assignment 1:
Suppose a consulting firm that offers security consultancy needs to obtain licenses for n
different pieces of cryptographic software. Due to regulations, they can buy only one license
per month. Each license currently costs Rs 1000, but the cost of each license increases at an
exponential rate rj (> 1) every month. For example, if license j is purchased t months from now,
it will cost 1000 * r_j^t (or base_cost * r_j^t). All price growth rates are distinct. Given the
rates of price growth, determine the order in which licenses should be bought to minimize total
cost. Implement this in a program.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class License:
    name: str
    rate: float
    base_cost: float = 1000.0


def optimal_license_schedule(
    licenses: List[License],
) -> Tuple[List[Tuple[int, License, float]], float]:
    """
    Determines the optimal purchase schedule to minimize the total cost of n licenses.

    Greedy Strategy:
        Sort licenses in descending order of their price growth rate r_j.
        Buy the license with the highest growth rate first (at t = 0), the second highest
        at t = 1, and so on.

    Proof of Optimality (Exchange Argument):
        Suppose in an optimal schedule OPT, license a is bought at month t and license b
        is bought at month t+1, but r_a < r_b.
        Cost in OPT for these two licenses:
            C_opt = base * r_a^t + base * r_b^(t+1)
        If we swap their purchase order (buy b at month t, and a at month t+1):
            C_swapped = base * r_b^t + base * r_a^(t+1)
        Difference:
            C_opt - C_swapped = base * (r_b^(t+1) - r_b^t - (r_a^(t+1) - r_a^t))
                              = base * (r_b^t * (r_b - 1) - r_a^t * (r_a - 1))
        Since r_b > r_a > 1 and t >= 0, we have r_b^t >= r_a^t and (r_b - 1) > (r_a - 1) > 0.
        Thus, C_opt - C_swapped > 0, meaning swapping them strictly decreases the total cost.
        Therefore, any inversion can be swapped to reduce cost, proving that sorting by
        decreasing rate is strictly optimal.

    Complexity:
        Time Complexity: O(n log n) due to sorting.
        Space Complexity: O(n) to store schedule results.
    """
    # Sort licenses in descending order of growth rate
    sorted_licenses = sorted(licenses, key=lambda lic: lic.rate, reverse=True)

    schedule: List[Tuple[int, License, float]] = []
    total_cost = 0.0

    for month, lic in enumerate(sorted_licenses):
        cost_at_t = lic.base_cost * (lic.rate ** month)
        schedule.append((month, lic, cost_at_t))
        total_cost += cost_at_t

    return schedule, total_cost


if __name__ == "__main__":
    print("=" * 75)
    print("Assignment 1: Optimal Cryptographic Software License Purchase Schedule")
    print("=" * 75)

    sample_licenses = [
        License("AES Encryption Suite", rate=1.20, base_cost=1000.0),
        License("RSA Key Manager", rate=1.50, base_cost=1000.0),
        License("Post-Quantum Signer", rate=1.80, base_cost=1000.0),
        License("Elliptic Curve Tool", rate=1.10, base_cost=1000.0),
        License("Zero-Knowledge Prover", rate=1.35, base_cost=1000.0),
    ]

    schedule, total = optimal_license_schedule(sample_licenses)

    print(f"{'Month':<6} | {'License Name':<25} | {'Rate (r_j)':<12} | {'Cost (Rs.)':<15}")
    print("-" * 65)
    for month, lic, cost in schedule:
        print(f"Month {month:<2} | {lic.name:<25} | {lic.rate:<12.2f} | Rs. {cost:<12.2f}")
    print("-" * 65)
    print(f"Minimum Total Purchase Cost: Rs. {total:.2f}")

    # Verify that rates are strictly monotonically decreasing with time
    for i in range(len(schedule) - 1):
        assert schedule[i][1].rate > schedule[i + 1][1].rate, "Ordering violation!"
    print("\nVerification Passed: Highest growth rates purchased earliest!")
    print("=" * 75)
