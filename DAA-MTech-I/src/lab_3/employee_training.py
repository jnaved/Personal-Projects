'''
Problem 2 - Employee Training Planning
An organisation needs to train an employee in a sequence of training modules. Each module
has a training cost and provides a certain benefit to the employee.
The employee has a maximum training budget of B. From the available modules, each module
can either be selected once or skipped. The objective is to select a set of modules whose
total cost does not exceed B while maximising the total benefit.

This is the 0/1 Knapsack Problem formulation.

Analysis Extension:
Unbounded Employee Training Planning (where each module can be selected multiple times).
'''

'''
Budget Capacity : B
Module Costs    : costs = [c1, c2, ..., cn]
Module Benefits : benefits = [b1, b2, ..., bn]
Objective       : Maximize total benefit such that total cost <= B.

0/1 Knapsack DP Recurrence:
    dp[i][j] is the maximum benefit achievable using a subset of the first i modules with budget j.

    if j < costs[i-1]:
        dp[i][j] = dp[i-1][j]
    else:
        dp[i][j] = max(dp[i-1][j], dp[i-1][j - costs[i-1]] + benefits[i-1])

Final Answer: dp[n][B]
Time Complexity: O(n * B)
Space Complexity: O(n * B) (or O(B) space-optimized)
'''

import time
import random
from pathlib import Path
import matplotlib.pyplot as plt


def implement_recursion_employee_training(idx: int, budget: int, costs: list[int], benefits: list[int]) -> int:
    """
    Pure recursive 0/1 knapsack approach (Exhaustive search).
    """
    if idx == 0 or budget <= 0:
        return 0

    cost = costs[idx - 1]
    benefit = benefits[idx - 1]

    if cost > budget:
        return implement_recursion_employee_training(idx - 1, budget, costs, benefits)
    else:
        exclude_module = implement_recursion_employee_training(idx - 1, budget, costs, benefits)
        include_module = implement_recursion_employee_training(idx - 1, budget - cost, costs, benefits) + benefit
        return max(exclude_module, include_module)


def implement_01_employee_training(
    budget: int,
    costs: list[int],
    benefits: list[int],
    module_names: list[str] | None = None
) -> tuple[int, list[str], list[list[int]]]:
    """
    0/1 Knapsack Dynamic Programming solution for Employee Training Planning.
    Returns:
        (max_benefit, selected_modules, dp_table)
    """
    n = len(costs)
    if module_names is None:
        module_names = [f"M{i+1}" for i in range(n)]

    # dp[i][j]: max benefit with first i modules and budget j
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        c = costs[i - 1]
        b = benefits[i - 1]
        for j in range(1, budget + 1):
            if j < c:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - c] + b)

    # Backtracking to reconstruct selected modules
    selected_modules = []
    curr_budget = budget
    for i in range(n, 0, -1):
        # If value comes from dp[i-1][curr_budget - cost] + benefit, module was selected
        if dp[i][curr_budget] != dp[i - 1][curr_budget]:
            selected_modules.append(module_names[i - 1])
            curr_budget -= costs[i - 1]

    selected_modules.reverse()
    return dp[n][budget], selected_modules, dp


def implement_unbounded_employee_training(
    budget: int,
    costs: list[int],
    benefits: list[int],
    module_names: list[str] | None = None
) -> tuple[int, list[str], list[list[int]]]:
    """
    Unbounded Dynamic Programming solution (each module can be selected multiple times).
    Used for the Analysis section.
    Returns:
        (max_benefit, selected_modules, dp_table)
    """
    n = len(costs)
    if module_names is None:
        module_names = [f"M{i+1}" for i in range(n)]

    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        c = costs[i - 1]
        b = benefits[i - 1]
        for j in range(1, budget + 1):
            if j < c:
                dp[i][j] = dp[i - 1][j]
            else:
                # Same row dp[i][j-c] allows selecting the module multiple times
                dp[i][j] = max(dp[i - 1][j], dp[i][j - c] + b)

    # Reconstruction for unbounded case
    selected_modules = []
    curr_budget = budget
    curr_i = n
    while curr_i > 0 and curr_budget > 0:
        c = costs[curr_i - 1]
        b = benefits[curr_i - 1]
        if curr_budget >= c and dp[curr_i][curr_budget] == dp[curr_i][curr_budget - c] + b:
            selected_modules.append(module_names[curr_i - 1])
            curr_budget -= c
        else:
            curr_i -= 1

    selected_modules.reverse()
    return dp[n][budget], selected_modules, dp


def generate_synthetic_modules(count: int) -> tuple[list[int], list[int], list[str]]:
    """Generates synthetic costs, benefits, and module names."""
    random.seed(42 + count)
    costs = []
    benefits = []
    names = [f"M{i+1}" for i in range(count)]
    for _ in range(count):
        c = random.randint(2, 12)
        b = c + random.randint(1, 8)
        costs.append(c)
        benefits.append(b)
    return costs, benefits, names


def run_static_demonstration():
    """
    Runs the static example from the problem statement:
    Modules:
      M1: Cost = 4, Benefit = 7
      M2: Cost = 6, Benefit = 10
      M3: Cost = 5, Benefit = 8
      M4: Cost = 3, Benefit = 5
    """
    print("\n" + "=" * 80)
    print("      PROBLEM 2: EMPLOYEE TRAINING PLANNING -- STATIC DEMONSTRATION")
    print("=" * 80)

    module_names = ["M1", "M2", "M3", "M4"]
    costs = [4, 6, 5, 3]
    benefits = [7, 10, 8, 5]
    budgets = [8, 10, 12]

    print("\n[Input Dataset]")
    print(f"{'Module':<10} | {'Cost':<10} | {'Benefit':<10} | {'Benefit/Cost Ratio':<18}")
    print("-" * 55)
    for name, c, b in zip(module_names, costs, benefits):
        print(f"{name:<10} | {c:<10} | {b:<10} | {b/c:<18.2f}")

    for b_val in budgets:
        print(f"\n" + "-" * 75)
        print(f"  TARGET TRAINING BUDGET: B = {b_val}")
        print("-" * 75)

        # 1. 0/1 Knapsack (Original Problem)
        max_b_01, sel_01, dp_01 = implement_01_employee_training(
            budget=b_val, costs=costs, benefits=benefits, module_names=module_names
        )
        total_cost_01 = sum(costs[module_names.index(m)] for m in sel_01)

        print(f"  [0/1 Knapsack - Original Formulation]")
        print(f"    Max Benefit Achieved : {max_b_01}")
        print(f"    Selected Modules     : {sel_01}")
        print(f"    Total Cost Incurred  : {total_cost_01} / {b_val}")

        # 2. Unbounded Knapsack (Analysis Extension)
        max_b_unb, sel_unb, dp_unb = implement_unbounded_employee_training(
            budget=b_val, costs=costs, benefits=benefits, module_names=module_names
        )
        total_cost_unb = sum(costs[module_names.index(m)] for m in sel_unb)

        print(f"  [Unbounded Knapsack - Analysis Modification (Repeated Modules Allowed)]")
        print(f"    Max Benefit Achieved : {max_b_unb}")
        print(f"    Selected Modules     : {sel_unb}")
        print(f"    Total Cost Incurred  : {total_cost_unb} / {b_val}")

        if max_b_unb > max_b_01:
            print(f"    Insight              : Unbounded selection improves benefit by +{max_b_unb - max_b_01} points!")
        else:
            print(f"    Insight              : Both 0/1 and Unbounded yield the same optimal benefit.")

    print("=" * 80 + "\n")


def run_experiment_and_plot(budgets: list[int] = [15, 20, 25], module_counts: list[int] = [5, 10, 20]):
    """
    Empirical time complexity analysis of 0/1 Employee Training Planning
    across varying budgets and module counts.
    """
    module_data = {n: generate_synthetic_modules(n) for n in module_counts}

    dp_times = {n: [] for n in module_counts}
    rec_times = {n: [] for n in module_counts}

    print(f"\n{'Budget (B)':<10} | {'# Modules (N)':<14} | {'DP Time (ms)':<14} | {'Rec Time (ms)':<14} | {'Max Benefit':<12}")
    print("-" * 75)

    for budget in budgets:
        for n in module_counts:
            costs, benefits, names = module_data[n]

            # Measure DP
            t0 = time.perf_counter()
            max_b, _, _ = implement_01_employee_training(budget=budget, costs=costs, benefits=benefits, module_names=names)
            dp_t = (time.perf_counter() - t0) * 1000
            dp_times[n].append(dp_t)

            # Measure Recursion (safe for n <= 20 and small budget)
            if n <= 10:
                t1 = time.perf_counter()
                rec_b = implement_recursion_employee_training(idx=n, budget=budget, costs=costs, benefits=benefits)
                rec_t = (time.perf_counter() - t1) * 1000
            else:
                rec_t = float('nan')
                rec_b = "N/A"
            rec_times[n].append(rec_t)

            rec_t_str = f"{rec_t:.4f}" if rec_t == rec_t else "N/A"
            print(f"{budget:<10} | {n:<14} | {dp_t:<14.4f} | {rec_t_str:<14} | {max_b:<12}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {5: "#1f77b4", 10: "#ff7f0e", 20: "#2ca02c"}
    markers = {5: "o", 10: "s", 20: "^"}

    # Subplot 1: Budget vs Execution Time (DP & Recursion)
    for n in module_counts:
        ax1.plot(
            budgets, dp_times[n],
            label=f"DP (N={n})",
            color=colors[n],
            linestyle="-",
            marker=markers[n],
            linewidth=2,
            markersize=6
        )
        if any(t == t for t in rec_times[n]):
            ax1.plot(
                budgets, rec_times[n],
                label=f"Recursion (N={n})",
                color=colors[n],
                linestyle="--",
                marker=markers[n],
                linewidth=2,
                markersize=6
            )

    ax1.set_title("Training Budget (B) vs Execution Time", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Budget (B)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms)", fontsize=11)
    ax1.set_xticks(budgets)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Subplot 2: Number of Modules vs DP Execution Time
    b_colors = ["#9467bd", "#8c564b", "#e377c2"]
    b_markers = ["o", "D", "v"]
    for idx, budget in enumerate(budgets):
        dp_by_n = [dp_times[n][idx] for n in module_counts]
        ax2.plot(
            module_counts, dp_by_n,
            label=f"DP (Budget={budget})",
            color=b_colors[idx % len(b_colors)],
            marker=b_markers[idx % len(b_markers)],
            linewidth=2,
            markersize=7
        )

    ax2.set_title("Number of Modules (N) vs DP Execution Time", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Modules (N)", fontsize=11)
    ax2.set_ylabel("Execution Time (ms)", fontsize=11)
    ax2.set_xticks(module_counts)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.suptitle("Employee Training Planning (0/1 Knapsack): Empirical Time Complexity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save figure
    current_dir = Path(__file__).resolve().parent
    images_dir = current_dir.parent.parent / "images"
    save_path = images_dir / "employee_training_tc_analysis.png" if images_dir.exists() else current_dir / "employee_training_tc_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Plot Saved] Figure saved to: {save_path}")

    # Also sync with overleaf if present
    overleaf_images_dir = current_dir.parent.parent / "overleaf" / "lab_3" / "images"
    if overleaf_images_dir.exists():
        plt.savefig(overleaf_images_dir / "employee_training_tc_analysis.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    print("\nSelect Mode:")
    print("1. User Custom Input (Test 0/1 Knapsack, Unbounded, and Recursion)")
    print("2. Run Static Problem 2 Demonstration (0/1 Knapsack vs Unbounded)")
    print("3. Run Empirical Time Complexity Experiment & Plotting")

    choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip()

    if choice == '1':
        budget = int(input("Enter maximum training budget (B): "))
        n = int(input("Enter number of modules (N): "))
        module_names = []
        costs = []
        benefits = []
        for i in range(n):
            name = input(f"Enter name for module {i+1} [default M{i+1}]: ").strip() or f"M{i+1}"
            c = int(input(f"  Enter cost for {name}: "))
            b = int(input(f"  Enter benefit for {name}: "))
            module_names.append(name)
            costs.append(c)
            benefits.append(b)

        t = time.perf_counter()
        max_b_01, sel_01, _ = implement_01_employee_training(budget, costs, benefits, module_names)
        dp_01_time = (time.perf_counter() - t) * 1000
        print(f"\n[0/1 DP]     Max Benefit: {max_b_01}, Selected Modules: {sel_01}, Time: {dp_01_time:.4f} ms")

        t = time.perf_counter()
        max_b_unb, sel_unb, _ = implement_unbounded_employee_training(budget, costs, benefits, module_names)
        dp_unb_time = (time.perf_counter() - t) * 1000
        print(f"[Unbounded]  Max Benefit: {max_b_unb}, Selected Modules: {sel_unb}, Time: {dp_unb_time:.4f} ms")

        if n <= 20:
            t = time.perf_counter()
            max_b_rec = implement_recursion_employee_training(n, budget, costs, benefits)
            rec_time = (time.perf_counter() - t) * 1000
            print(f"[Recursion]  Max Benefit: {max_b_rec}, Time: {rec_time:.4f} ms")

    elif choice == '3':
        print("\n--- Running Graphical Experiment (5, 10, 20 Modules x 3 Budgets) ---")
        run_experiment_and_plot()
    else:
        run_static_demonstration()
