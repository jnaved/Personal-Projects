"""
========================================================================================
DAA Lab 11 - Assignment 3: Currency Exchange Arbitrage Detection Using Bellman–Ford
========================================================================================

Problem Statement:
------------------
A financial platform receives exchange rates between different currencies. It must detect
whether a sequence of currency exchanges can yield a profit by converting money through a
cycle (arbitrage).

Mathematical Modeling:
----------------------
Let R(u, v) be the exchange rate from currency u to currency v (1 unit of u buys R(u, v) of v).
A sequence of exchanges around a cycle c_1 -> c_2 -> ... -> c_k -> c_1 yields a profit if:
    R(c_1, c_2) * R(c_2, c_3) * ... * R(c_k, c_1) > 1.0

Taking natural logarithm on both sides:
    ln(R(c_1, c_2)) + ln(R(c_2, c_3)) + ... + ln(R(c_k, c_1)) > 0

Multiplying by -1 reverses the inequality:
    (-ln(R(c_1, c_2))) + (-ln(R(c_2, c_3))) + ... + (-ln(R(c_k, c_1))) < 0

Therefore, if we assign edge weights:
    w(u, v) = -ln(R(u, v))
then:
    Any negative-weight cycle in this graph corresponds directly to an arbitrage opportunity!

Algorithm:
----------
1. Model currencies as vertices and currency conversions as directed edges with weight -ln(R).
2. Initialize dist[v] = 0 for all currencies (detects cycles across all components).
3. Relax all edges |V| - 1 times.
4. On the |V|-th pass, check for any edge (u, v) that can still be relaxed (dist[u] + w < dist[v]).
5. If found, trace predecessor pointers back |V| steps to enter the cycle, then follow predecessors
   until the cycle closes.
6. Reconstruct the currency conversion sequence, compute the net profit multiplier, and simulate
   the trade execution.
========================================================================================
"""

import math
from typing import Dict, List, Optional, Set, Tuple


class CurrencyExchangeMarket:
    """Represents a foreign exchange market network."""

    def __init__(self):
        self.currencies: Set[str] = set()
        # Edge format: (from_currency, to_currency, exchange_rate, weight)
        self.rates: Dict[Tuple[str, str], float] = {}
        self.edges: List[Tuple[str, str, float, float]] = []

    def add_exchange_rate(self, from_curr: str, to_curr: str, rate: float) -> None:
        """
        Add an exchange rate from from_curr to to_curr.
        Weight = -ln(rate).
        """
        if rate <= 0:
            raise ValueError("Exchange rate must be strictly positive.")
        self.currencies.add(from_curr)
        self.currencies.add(to_curr)
        self.rates[(from_curr, to_curr)] = rate
        weight = -math.log(rate)
        self.edges.append((from_curr, to_curr, rate, weight))

    def detect_arbitrage(self) -> Tuple[bool, Optional[List[str]], float, float]:
        """
        Uses Bellman-Ford algorithm to detect negative weight cycles.
        Returns:
            (has_arbitrage, cycle_path, net_multiplier, profit_percentage)
        """
        curr_list = sorted(list(self.currencies))
        n = len(curr_list)
        if n == 0:
            return False, None, 1.0, 0.0

        # Initialize distances to 0 for all vertices to detect negative cycles in any component
        dist: Dict[str, float] = {c: 0.0 for c in curr_list}
        parent: Dict[str, Optional[str]] = {c: None for c in curr_list}

        # Relax all edges |V| - 1 times
        eps = 1e-9
        for _ in range(n - 1):
            for u, v, rate, weight in self.edges:
                if dist[u] + weight < dist[v] - eps:
                    dist[v] = dist[u] + weight
                    parent[v] = u

        # Check for negative-weight cycle on the |V|-th iteration
        cycle_node = None
        for u, v, rate, weight in self.edges:
            if dist[u] + weight < dist[v] - eps:
                cycle_node = v
                break

        if cycle_node is None:
            return False, None, 1.0, 0.0

        # Trace back |V| times to guarantee entering the cycle
        curr = cycle_node
        for _ in range(n):
            curr = parent[curr]

        # Extract the cycle
        cycle_start = curr
        cycle = [cycle_start]
        curr = parent[cycle_start]
        while curr != cycle_start:
            cycle.append(curr)
            curr = parent[curr]
        cycle.append(cycle_start)
        cycle.reverse()

        # Compute net exchange multiplier along the cycle
        multiplier = 1.0
        for i in range(len(cycle) - 1):
            u_c = cycle[i]
            v_c = cycle[i + 1]
            multiplier *= self.rates[(u_c, v_c)]

        profit_percentage = (multiplier - 1.0) * 100.0
        return True, cycle, multiplier, profit_percentage

    def simulate_arbitrage_trade(self, cycle: List[str], initial_amount: float = 10000.0) -> None:
        """Simulates trading along the detected arbitrage cycle step-by-step."""
        print(f"\nSimulating Arbitrage Execution with Initial Capital: {initial_amount:,.2f} {cycle[0]}")
        print("-" * 75)
        print(f"{'Step':<6} | {'Trade':<16} | {'Exchange Rate':<14} | {'Resulting Amount':<22}")
        print("-" * 75)

        current_amount = initial_amount
        for i in range(len(cycle) - 1):
            from_c = cycle[i]
            to_c = cycle[i + 1]
            rate = self.rates[(from_c, to_c)]
            next_amount = current_amount * rate
            trade_str = f"{from_c} -> {to_c}"
            print(f"Step {i + 1:<2} | {trade_str:<16} | {rate:<14.5f} | {next_amount:>14,.2f} {to_c}")
            current_amount = next_amount

        net_profit = current_amount - initial_amount
        profit_pct = (net_profit / initial_amount) * 100.0
        print("-" * 75)
        print(f"Final Amount : {current_amount:,.2f} {cycle[0]}")
        print(f"Net Profit   : {net_profit:,.2f} {cycle[0]} (+{profit_pct:.2f}%)")


# ========================================================================================
# Demonstration / Driver Code
# ========================================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ASSIGNMENT 3: CURRENCY EXCHANGE ARBITRAGE DETECTION USING BELLMAN-FORD")
    print("=" * 80)

    # -----------------------------------------------------------------------------------
    # Scenario 1: Market WITH Arbitrage Opportunity
    # -----------------------------------------------------------------------------------
    print("\n[SCENARIO 1] Foreign Exchange Market with Inefficiencies (Arbitrage Exists)")
    print("-" * 80)
    market_with_arb = CurrencyExchangeMarket()

    # Exchange rates:
    # 1 USD -> 0.92 EUR
    # 1 EUR -> 0.86 GBP
    # 1 GBP -> 190.0 JPY
    # 1 JPY -> 0.0068 USD
    # Product = 0.92 * 0.86 * 190.0 * 0.0068 = 1.0222144 (2.22% profit per cycle!)
    rates_data = [
        ("USD", "EUR", 0.92),
        ("EUR", "USD", 1.08),
        ("EUR", "GBP", 0.86),
        ("GBP", "EUR", 1.15),
        ("GBP", "JPY", 190.0),
        ("JPY", "GBP", 0.0052),
        ("JPY", "USD", 0.0068),
        ("USD", "JPY", 145.0),
        ("USD", "CAD", 1.35),
        ("CAD", "USD", 0.74),
    ]

    for u, v, r in rates_data:
        market_with_arb.add_exchange_rate(u, v, r)

    print("Market Exchange Rates and Logarithmic Weights (-ln(Rate)):")
    for u, v, r, w in market_with_arb.edges[:6]:
        print(f"  {u} -> {v} : Rate = {r:<9.4f} | Edge Weight = {w:+.5f}")
    print("  ... (and other currency pairs)")

    has_arb, cycle, multiplier, profit_pct = market_with_arb.detect_arbitrage()

    if has_arb and cycle is not None:
        print("\n>>> ARBITRAGE DETECTED! <<<")
        print(f"Arbitrage Cycle     : {' -> '.join(cycle)}")
        print(f"Effective Multiplier: {multiplier:.6f}")
        print(f"Theoretical Profit  : +{profit_pct:.2f}%")
        market_with_arb.simulate_arbitrage_trade(cycle, initial_amount=10000.0)
    else:
        print("\nNo arbitrage opportunity found in Scenario 1.")

    # -----------------------------------------------------------------------------------
    # Scenario 2: Efficient / Balanced Market (NO Arbitrage)
    # -----------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[SCENARIO 2] Perfectly Efficient Market (No Arbitrage)")
    print("-" * 80)
    market_efficient = CurrencyExchangeMarket()

    # Perfectly consistent exchange rates (no negative cycles)
    market_efficient.add_exchange_rate("USD", "EUR", 0.90)
    market_efficient.add_exchange_rate("EUR", "USD", 1.0 / 0.90)
    market_efficient.add_exchange_rate("EUR", "GBP", 0.80)
    market_efficient.add_exchange_rate("GBP", "EUR", 1.0 / 0.80)
    market_efficient.add_exchange_rate("GBP", "USD", 1.0 / (0.90 * 0.80))
    market_efficient.add_exchange_rate("USD", "GBP", 0.90 * 0.80)

    has_arb2, cycle2, _, _ = market_efficient.detect_arbitrage()
    if has_arb2:
        print(f"Arbitrage found: {' -> '.join(cycle2)}")
    else:
        print("Result: No negative-weight cycle detected.")
        print("Market is efficient; no risk-free arbitrage profit is possible.")
    print("=" * 80)
