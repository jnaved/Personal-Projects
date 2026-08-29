'''
You are given a set of coin denominations and a target amount. Each denomination can be used
any number of times. Determine the minimum number of coins required to obtain the target
amount. If the target amount cannot be formed using the given denominations, report that no
solution exists.
'''

'''
Maximum Coin Capacity : A
Coin Denomination Value : Ci
Each coin is counted as 1 coin
Objective is to minimize the total number of coins to achieve the total capacity A.


dp[i][j] is the minimum number of coins required to achieve total coin value j (0<=j<=A) by
using the first i coin denominations({c1, c2, ..., cn})

Recurrence Relation

    When coin is excluded(or cannot be taken), the minimum number of coins required is the value
    directly above it, i.e., dp[i-1][j]

    When coin is included, picking the coin means an amount of j-ci remains and thus the relation 
    becomes (dp[i][j-1] + 1) (Same row as the coin can be picked multiple amount of times)

    Combinedly, the relation becomes

    if j < ci:
        dp[i][j] = dp[i-1][j]
    else:
        dp[i][j] = min(dp[i-1][j], dp[i][j-ci] + 1)


The Final Answer stays on dp[N][A].
Time Complexity is O(N*A)
Space Complexity is O(N*A)
'''

import random
import time
from pathlib import Path
import matplotlib.pyplot as plt


def implement_recursion_coin_change(amount:int, coin_denominations: list[int]):
    if amount == 0:
        return 0
    if amount < 0:
        return float('inf')

    min_coins = float('inf')
    for coin in coin_denominations:
        result = implement_recursion_coin_change(amount=amount-coin, coin_denominations=coin_denominations)
        if result != float('inf'):
            min_coins = min(min_coins, result + 1)

    return min_coins


def implement_coin_change_problem(amount: int, coin_denominations: list[int]):
    n = len(coin_denominations)

    dp = [[float('inf')] * (amount + 1) for _ in range(n + 1)]

    # base condition
    # ways to make amount 0 is 0( not take anything )
    for i in range(n+1):
        dp[i][0] = 0

    for i in range(1, n+1):
        coin_denomination = coin_denominations[i-1]
        for j in range(1, amount+1):
            if j < coin_denomination:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-coin_denomination] + 1)
    
    return dp[n][amount]


def generate_denominations(count: int) -> list[int]:
    """Generates meaningful coin denominations including 1."""
    standard_pool = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000]
    if count <= len(standard_pool):
        return sorted(set([1] + standard_pool[:count]))
    
    coins = standard_pool.copy()
    curr = 2500
    while len(coins) < count:
        coins.append(curr)
        curr += 500
    return sorted(coins[:count])


def run_experiment_and_plot(amounts: list[int] = [15, 20, 25], denom_counts: list[int] = [5, 10, 20]):
    """
    Runs time-complexity experiment across different amounts and denomination counts,
    and plots the execution times graphically.
    """
    denom_sets = {count: generate_denominations(count) for count in denom_counts}
    
    dp_times = {count: [] for count in denom_counts}
    rec_times = {count: [] for count in denom_counts}

    print(f"\n{'Amount':<8} | {'# Coins (N)':<12} | {'DP Time (ms)':<14} | {'Rec Time (ms)':<14} | {'Min Coins':<10}")
    print("-" * 75)

    for amount in amounts:
        for count in denom_counts:
            coins = denom_sets[count]
            
            # Measure DP
            t0 = time.perf_counter()
            dp_ans = implement_coin_change_problem(amount=amount, coin_denominations=coins)
            dp_t = (time.perf_counter() - t0) * 1000
            dp_times[count].append(dp_t)

            # Measure Recursion (safe for amounts <= 25)
            if amount <= 25:
                t1 = time.perf_counter()
                rec_ans = implement_recursion_coin_change(amount=amount, coin_denominations=coins)
                rec_t = (time.perf_counter() - t1) * 1000
            else:
                rec_t = float('nan')
                rec_ans = "N/A"
            rec_times[count].append(rec_t)

            rec_t_str = f"{rec_t:.4f}" if rec_t == rec_t else "N/A"
            print(f"{amount:<8} | {count:<12} | {dp_t:<14.4f} | {rec_t_str:<14} | {dp_ans:<10}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {5: "#1f77b4", 10: "#ff7f0e", 20: "#2ca02c"}
    markers = {5: "o", 10: "s", 20: "^"}

    # Subplot 1: Target Amount vs Execution Time (DP & Recursion)
    for count in denom_counts:
        ax1.plot(
            amounts, dp_times[count],
            label=f"DP (N={count})",
            color=colors[count],
            linestyle="-",
            marker=markers[count],
            linewidth=2,
            markersize=6
        )
        if any(t == t for t in rec_times[count]):
            ax1.plot(
                amounts, rec_times[count],
                label=f"Recursion (N={count})",
                color=colors[count],
                linestyle="--",
                marker=markers[count],
                linewidth=2,
                markersize=6
            )

    ax1.set_title("Target Amount vs Execution Time (Log Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Target Amount (A)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms) - Log Scale", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_xticks(amounts)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Subplot 2: Number of Denominations vs Execution Time for each Amount (DP Focus)
    amt_colors = ["#9467bd", "#8c564b", "#e377c2"]
    amt_markers = ["o", "D", "v"]
    for idx, amount in enumerate(amounts):
        dp_by_denom = [dp_times[c][idx] for c in denom_counts]
        ax2.plot(
            denom_counts, dp_by_denom,
            label=f"DP (Amount={amount})",
            color=amt_colors[idx % len(amt_colors)],
            marker=amt_markers[idx % len(amt_markers)],
            linewidth=2,
            markersize=7
        )

    ax2.set_title("Number of Denominations vs DP Execution Time", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Denominations (N)", fontsize=11)
    ax2.set_ylabel("Execution Time (ms)", fontsize=11)
    ax2.set_xticks(denom_counts)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.suptitle("Coin Change Problem: Empirical Time Complexity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save figure
    current_dir = Path(__file__).resolve().parent
    images_dir = current_dir.parent.parent / "images"
    save_path = images_dir / "coin_change_tc_analysis.png" if images_dir.exists() else current_dir / "coin_change_tc_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Plot Saved] Figure saved to: {save_path}")

    # Also sync with overleaf if present
    overleaf_images_dir = current_dir.parent.parent / "overleaf" / "lab_3" / "images"
    if overleaf_images_dir.exists():
        plt.savefig(overleaf_images_dir / "coin_change_tc_analysis.png", dpi=300)

    plt.show()


def implement_greedy_coin_change(amount: int, coin_denominations: list[int]) -> tuple[int | float, list[int]]:
    """
    Greedy Strategy: Always choose the largest possible denomination at every step.
    Returns:
        (total_coins, list_of_chosen_coins)
        If the amount cannot be formed, returns (float('inf'), []).
    """
    sorted_denoms = sorted(coin_denominations, reverse=True)
    remaining = amount
    chosen_coins = []

    for coin in sorted_denoms:
        if remaining <= 0:
            break
        count = remaining // coin
        if count > 0:
            chosen_coins.extend([coin] * count)
            remaining -= coin * count

    if remaining == 0:
        return len(chosen_coins), chosen_coins
    else:
        return float('inf'), []


def run_static_counterexamples():
    """
    Demonstrates static test cases and counterexamples comparing Greedy vs Dynamic Programming.
    """
    print("\n" + "=" * 80)
    print("      GREEDY VS. DYNAMIC PROGRAMMING: STATIC COUNTEREXAMPLES ANALYSIS")
    print("=" * 80)

    test_cases = [
        {
            "title": "Counterexample 1: Suboptimal Coin Count",
            "denominations": [1, 3, 4],
            "amount": 6,
            "description": "Greedy picks largest coin 4 first, leaving remainder 2 (forced to pick 1+1 = 3 coins).\nDP optimally picks two 3-unit coins (2 coins)."
        },
        {
            "title": "Counterexample 2: Complete Greedy Failure (No Solution Found)",
            "denominations": [3, 4],
            "amount": 6,
            "description": "Greedy picks 4, leaving remainder 2 with no valid coin <= 2 (fails).\nDP finds valid optimal solution: 3 + 3 = 6 (2 coins)."
        },
        {
            "title": "Counterexample 3: Canonical System (Standard Indian Denominations)",
            "denominations": [1, 2, 5, 10, 20, 50, 100],
            "amount": 43,
            "description": "Canonical currency system where Greedy and DP both yield the global optimum."
        }
    ]

    for idx, tc in enumerate(test_cases, 1):
        denoms = tc["denominations"]
        amt = tc["amount"]

        greedy_count, greedy_coins = implement_greedy_coin_change(amount=amt, coin_denominations=denoms)
        dp_count = implement_coin_change_problem(amount=amt, coin_denominations=denoms)

        print(f"\n[Case {idx}] {tc['title']}")
        print(f"  Denominations : {denoms}")
        print(f"  Target Amount : {amt}")
        print(f"  Note          : {tc['description']}")
        print(f"  --------------------------------------------------------------")
        greedy_res_str = f"{greedy_count} coins {greedy_coins}" if greedy_count != float('inf') else "FAILED (No solution found)"
        print(f"  Greedy Result : {greedy_res_str}")
        print(f"  DP Result     : {dp_count} coins (Optimal)")
        
        if greedy_count == float('inf'):
            print(f"  Conclusion    : Greedy FAILS completely; DP SUCCEEDS.")
        elif greedy_count > dp_count:
            print(f"  Conclusion    : Greedy is SUBOPTIMAL ({greedy_count} > {dp_count}); DP is OPTIMAL.")
        else:
            print(f"  Conclusion    : Both Greedy and DP produce optimal solution ({greedy_count} == {dp_count}).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\nSelect Mode:")
    print("1. User Custom Input (Test Greedy, DP, and Recursion)")
    print("2. Run Static Counterexamples & Greedy vs DP Analysis")
    print("3. Run Empirical Time Complexity Experiment & Plotting")
    
    choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip()
    
    if choice == '1':
        amount = int(input("Enter the max coin amount: "))
        print("Enter the coin denominations (space-separated): ")
        coin_denominations = list(map(int, input().split()))
    
        t = time.perf_counter()
        greedy_ans, greedy_coins = implement_greedy_coin_change(amount=amount, coin_denominations=coin_denominations)
        greedy_time = (time.perf_counter() - t) * 1000
        print(f"\n[Greedy] Min coins: {greedy_ans} (Coins: {greedy_coins}), Time: {greedy_time:.4f} ms")

        t = time.perf_counter()
        min_coins_dp = implement_coin_change_problem(amount=amount, coin_denominations=coin_denominations)
        dp_time = (time.perf_counter() - t) * 1000
        print(f"[DP]     Min coins: {min_coins_dp}, Time: {dp_time:.4f} ms")

        if amount <= 30:
            t = time.perf_counter()
            min_coins_rec = implement_recursion_coin_change(amount=amount, coin_denominations=coin_denominations)
            rec_time = (time.perf_counter() - t) * 1000
            print(f"[Rec]    Min coins: {min_coins_rec}, Time: {rec_time:.4f} ms")
        else:
            print("[Rec]    Skipped (Amount too large for exponential recursion)")

    elif choice == '3':
        print("\n--- Running Graphical Experiment (5, 10, 20 Denominations x 3 Target Amounts) ---")
        run_experiment_and_plot()
    else:
        run_static_counterexamples()