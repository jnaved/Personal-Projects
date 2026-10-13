"""
N-Queens Problem (N = 8) with Constraints & Search Statistics
=============================================================
DAA Lab 8.1 - Question 1.1

Problem Description:
--------------------
Solve the 8-Queens problem on an 8x8 chessboard such that no two queens
attack each other (no two queens in the same row, column, or diagonal).

Extensions in this program:
  1. Forbidden Cells: Cells where queens cannot be placed.
  2. Pre-Placed Queens: Queens whose positions are fixed in advance.
  3. All Solutions: Prints every valid arrangement and counts total solutions.
  4. Contradiction Rejection: Rejects invalid inputs (e.g., pre-placed queens
     attacking each other, or placing a queen on a forbidden square).
  5. Immediate Pruning: Backtracking cuts off a branch as soon as a queen
     violates any attack or forbidden constraint (O(1) safety checks).
  6. Symmetry Reduction: Avoids exploring mirror-image duplicate arrangements
     when constraints allow it.
  7. Search Statistics: Counts nodes explored, backtracks, and execution time.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Type aliases for easy reading
Coordinate = Tuple[int, int]
Cell = Union[str, Coordinate]


# =====================================================================
# 1. HELPER FUNCTIONS: COORDINATE CONVERSION & BOARD DISPLAY
# =====================================================================

def to_coord(cell: Cell, n: int = 8) -> Coordinate:
    """
    Converts a cell like 'e4' or (4, 4) into a (row, col) tuple.
      - In chess notation: 'a' to 'h' are columns 0 to 7.
      - Rank 8 is row 0 (top), Rank 1 is row 7 (bottom).
    """
    if isinstance(cell, (tuple, list)):
        r, c = int(cell[0]), int(cell[1])
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(f"Coordinate ({r}, {c}) is outside the {n}x{n} board.")
        return (r, c)

    if isinstance(cell, str):
        clean = cell.strip().lower()
        col = ord(clean[0]) - ord('a')
        rank = int(clean[1:])
        row = n - rank  # rank 8 -> row 0, rank 1 -> row 7
        if not (0 <= col < n and 1 <= rank <= n):
            raise ValueError(f"Cell '{cell}' is outside the {n}x{n} board.")
        return (row, col)

    raise ValueError(f"Invalid cell format: {cell}")


def to_notation(row: int, col: int, n: int = 8) -> str:
    """Converts (row, col) to algebraic chess notation, e.g. (4, 4) -> 'e4'."""
    file_letter = chr(ord('a') + col)
    rank = n - row
    return f"{file_letter}{rank}"


def print_board(solution: List[int], pre_placed: Set[Coordinate], forbidden: Set[Coordinate], n: int = 8) -> None:
    """
    Prints a clear visual chessboard in the terminal:
      'Q' = Placed Queen
      'P' = Pre-placed (fixed) Queen
      'X' = Forbidden Cell
      '.' = Empty square
    """
    print("   " + " ".join(chr(ord('a') + c) for c in range(n)))
    print("  +" + "--" * n + "+")
    for r in range(n):
        row_cells = []
        for c in range(n):
            if solution[r] == c:
                row_cells.append("P" if (r, c) in pre_placed else "Q")
            elif (r, c) in forbidden:
                row_cells.append("X")
            else:
                row_cells.append(".")
        rank = n - r
        print(f"{rank} |" + " ".join(row_cells) + f"| {rank}")
    print("  +" + "--" * n + "+")
    print("   " + " ".join(chr(ord('a') + c) for c in range(n)))
    print("  Legend: [Q] Queen  [P] Pre-placed Queen  [X] Forbidden  [.] Empty\n")


# =====================================================================
# 2. CONTRADICTION CHECKING (INPUT VALIDATION)
# =====================================================================

def validate_input(
    n: int,
    pre_placed: Set[Coordinate],
    forbidden: Set[Coordinate]
) -> None:
    """
    Checks for impossible or contradictory inputs and raises ValueError if invalid:
      1. Pre-placed queen placed on a forbidden square.
      2. Two pre-placed queens attacking each other (same row, column, or diagonal).
      3. A row has every column forbidden and no pre-placed queen.
    """
    # Check 1: Pre-placed queen on a forbidden square
    for r, c in pre_placed:
        if (r, c) in forbidden:
            cell_name = to_notation(r, c, n)
            raise ValueError(f"Contradiction: Pre-placed queen at '{cell_name}' occupies a forbidden cell!")

    # Check 2: Two pre-placed queens attacking each other
    queens = list(pre_placed)
    for i in range(len(queens)):
        r1, c1 = queens[i]
        for j in range(i + 1, len(queens)):
            r2, c2 = queens[j]
            name1 = to_notation(r1, c1, n)
            name2 = to_notation(r2, c2, n)

            if r1 == r2:
                raise ValueError(f"Contradiction: Queens at '{name1}' and '{name2}' share the same row!")
            if c1 == c2:
                raise ValueError(f"Contradiction: Queens at '{name1}' and '{name2}' share the same column!")
            if abs(r1 - r2) == abs(c1 - c2):
                raise ValueError(f"Contradiction: Queens at '{name1}' and '{name2}' attack each other diagonally!")

    # Check 3: Row where all columns are forbidden
    pre_placed_rows = {r for r, _ in pre_placed}
    for r in range(n):
        if r not in pre_placed_rows:
            forbidden_in_row = sum(1 for c in range(n) if (r, c) in forbidden)
            if forbidden_in_row == n:
                raise ValueError(f"Contradiction: Row {n - r} has all 8 cells forbidden; cannot place 8 queens!")


# =====================================================================
# 3. SAFETY CHECK (O(1) BOUNDING FUNCTION)
# =====================================================================

def is_safe(
    row: int,
    col: int,
    cols_used: Set[int],
    diag1_used: Set[int],  # row - col
    diag2_used: Set[int],  # row + col
    forbidden: Set[Coordinate]
) -> bool:
    """
    O(1) Bounding Function:
    Returns True if placing a queen at (row, col) is valid:
      - Cell is not forbidden
      - Column 'col' is not already occupied
      - Main diagonal (row - col) is not attacked
      - Anti-diagonal (row + col) is not attacked
    """
    if (row, col) in forbidden:
        return False
    if col in cols_used:
        return False
    if (row - col) in diag1_used:
        return False
    if (row + col) in diag2_used:
        return False
    return True


# =====================================================================
# 4. SYMMETRY REDUCTION HELPER
# =====================================================================

def has_vertical_symmetry(pre_placed: Set[Coordinate], forbidden: Set[Coordinate], n: int = 8) -> bool:
    """
    Checks if the problem constraints are symmetric under left-to-right mirror reflection:
    (r, c) <---> (r, n - 1 - c).
    If symmetric, the right half of the board is a mirror of the left half.
    """
    for r, c in pre_placed:
        if (r, n - 1 - c) not in pre_placed:
            return False
    for r, c in forbidden:
        if (r, n - 1 - c) not in forbidden:
            return False
    return True


# =====================================================================
# 5. CORE BACKTRACKING SOLVER
# =====================================================================

def solve_8_queens(
    n: int = 8,
    pre_placed: Optional[List[Cell]] = None,
    forbidden: Optional[List[Cell]] = None,
    use_symmetry_reduction: bool = False,
    find_all: bool = True
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    Solves the 8-Queens problem using recursive backtracking.

    Parameters:
      n: Board size (default 8).
      pre_placed: List of pre-placed queen cells (e.g. ['e4', 'd1']).
      forbidden: List of forbidden cells (e.g. ['a1', 'h8']).
      use_symmetry_reduction: If True, avoids searching mirror duplicates when allowed.
      find_all: If True, finds all arrangements; if False, stops after the first one.

    Returns:
      (solutions, stats):
        - solutions: List of boards, where each board is a list of length n,
                     and board[row] = column index of the queen in that row.
        - stats: Dictionary containing search statistics.
    """
    # 1. Parse cell inputs into sets of (r, c) tuples
    pre_placed_set: Set[Coordinate] = {to_coord(c, n) for c in (pre_placed or [])}
    forbidden_set: Set[Coordinate] = {to_coord(c, n) for c in (forbidden or [])}

    # 2. Validate input and reject contradictions
    validate_input(n, pre_placed_set, forbidden_set)

    # 3. Map fixed row -> fixed column for quick lookup
    fixed_row_to_col: Dict[int, int] = {r: c for r, c in pre_placed_set}

    # 4. State tracking sets for O(1) conflict detection
    cols_used: Set[int] = set()
    diag1_used: Set[int] = set()  # row - col
    diag2_used: Set[int] = set()  # row + col

    # Register already pre-placed queens into collision structures
    for r, c in pre_placed_set:
        cols_used.add(c)
        diag1_used.add(r - c)
        diag2_used.add(r + c)

    # 5. Search statistics
    stats: Dict[str, Any] = {
        "nodes_explored": 0,
        "backtracks": 0,
        "pruned_conflicts": 0,
        "pruned_forbidden": 0,
        "solutions_found": 0,
        "symmetry_reduction_applied": False,
        "elapsed_time_ms": 0.0,
    }

    # Check if left-to-right mirror symmetry reduction can be safely applied
    can_reduce_symmetry = (
        use_symmetry_reduction
        and (0 not in fixed_row_to_col)
        and has_vertical_symmetry(pre_placed_set, forbidden_set, n)
    )
    stats["symmetry_reduction_applied"] = can_reduce_symmetry

    solutions: List[List[int]] = []
    board: List[int] = [-1] * n

    start_time = time.perf_counter()

    # 6. Recursive Backtracking Function
    def backtrack(row: int) -> bool:
        stats["nodes_explored"] += 1

        # BASE CASE: All 8 queens successfully placed!
        if row == n:
            solutions.append(list(board))
            stats["solutions_found"] += 1
            return not find_all  # return True to stop if find_all is False

        # CASE A: This row has a fixed pre-placed queen
        if row in fixed_row_to_col:
            board[row] = fixed_row_to_col[row]
            should_stop = backtrack(row + 1)
            board[row] = -1
            if should_stop:
                return True
            stats["backtracks"] += 1
            return False

        # CASE B: Normal unassigned row -> try placing a queen in column c
        # If symmetry reduction applies at row 0, only explore columns 0 to n//2 - 1
        cols_to_try = range(n // 2) if (row == 0 and can_reduce_symmetry) else range(n)

        for col in cols_to_try:
            # Check forbidden cell constraint
            if (row, col) in forbidden_set:
                stats["pruned_forbidden"] += 1
                continue

            # Check attack constraint
            if not is_safe(row, col, cols_used, diag1_used, diag2_used, forbidden_set):
                stats["pruned_conflicts"] += 1
                continue

            # --- PLACE QUEEN ---
            board[row] = col
            cols_used.add(col)
            diag1_used.add(row - col)
            diag2_used.add(row + col)

            # Recurse to next row
            should_stop = backtrack(row + 1)
            if should_stop:
                return True

            # --- BACKTRACK (UNDO MOVE) ---
            board[row] = -1
            cols_used.remove(col)
            diag1_used.remove(row - col)
            diag2_used.remove(row + col)
            stats["backtracks"] += 1

        return False

    # Start search from row 0
    backtrack(0)
    stats["elapsed_time_ms"] = (time.perf_counter() - start_time) * 1000

    return solutions, stats


def has_valid_arrangement(
    n: int = 8,
    pre_placed: Optional[List[Cell]] = None,
    forbidden: Optional[List[Cell]] = None
) -> bool:
    """Returns True if at least one valid arrangement exists, False otherwise."""
    try:
        solutions, _ = solve_8_queens(n, pre_placed, forbidden, find_all=False)
        return len(solutions) > 0
    except ValueError:
        return False


# =====================================================================
# 6. MAIN PROGRAM & LAB DEMONSTRATION
# =====================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("       DAA Lab 8.1 - Question 1.1: 8-Queens Problem Solver       ")
    print("=" * 72)

    # -----------------------------------------------------------------
    # Test 1: Standard Conventional 8-Queens (No extra constraints)
    # -----------------------------------------------------------------
    print("\n--- [Test 1: Standard 8-Queens (Unconstrained)] ---")
    sols_std, stats_std = solve_8_queens(n=8)
    print(f"Valid arrangement exists? : {len(sols_std) > 0}")
    print(f"Total valid arrangements  : {len(sols_std)} (Theoretical: 92)")
    print(f"Nodes explored            : {stats_std['nodes_explored']}")
    print(f"Execution time            : {stats_std['elapsed_time_ms']:.2f} ms")
    assert len(sols_std) == 92

    # -----------------------------------------------------------------
    # Test 2: Pre-placed Queens and Forbidden Cells (Question 1.1 Demo)
    # -----------------------------------------------------------------
    print("\n--- [Test 2: Pre-Placed Queen + Forbidden Cells] ---")
    # Queen fixed at 'e4' (row 4, col 4).
    # Some cells forbidden: 'a8', 'h8', 'a1', 'h1', 'c3'.
    my_pre_placed = ["e4"]
    my_forbidden = ["a8", "h8", "a1", "h1", "c3"]

    print(f"Pre-placed queens : {my_pre_placed}")
    print(f"Forbidden cells   : {my_forbidden}")

    sols_constrained, stats_c = solve_8_queens(
        n=8,
        pre_placed=my_pre_placed,
        forbidden=my_forbidden
    )

    print(f"Valid arrangement exists? : {len(sols_constrained) > 0}")
    print(f"Total valid arrangements  : {len(sols_constrained)}")
    print(f"Nodes explored            : {stats_c['nodes_explored']}")
    print(f"Pruned by attack check    : {stats_c['pruned_conflicts']}")
    print(f"Pruned by forbidden cells : {stats_c['pruned_forbidden']}")
    print(f"Execution time            : {stats_c['elapsed_time_ms']:.2f} ms")

    # Display the first valid arrangement on the board
    if sols_constrained:
        first_sol = sols_constrained[0]
        algebraic_coords = [to_notation(r, first_sol[r]) for r in range(8)]
        print(f"\nSample Arrangement #1 (Queen positions: {', '.join(algebraic_coords)}):")
        pre_set = {to_coord(c) for c in my_pre_placed}
        forb_set = {to_coord(c) for c in my_forbidden}
        print_board(first_sol, pre_set, forb_set)

    # -----------------------------------------------------------------
    # Test 3: Symmetry Reduction Demo
    # -----------------------------------------------------------------
    print("--- [Test 3: Symmetry Reduction (Mirror Pruning)] ---")
    # On an unconstrained board, left-to-right reflection reduces row 0 choices by half
    sols_sym, stats_sym = solve_8_queens(n=8, use_symmetry_reduction=True)
    print(f"Without symmetry reduction: {len(sols_std)} solutions ({stats_std['nodes_explored']} nodes explored)")
    print(f"With symmetry reduction   : {len(sols_sym)} non-mirror solutions ({stats_sym['nodes_explored']} nodes explored)")
    print(f"Notice: Explored space was halved ({stats_sym['nodes_explored']} vs {stats_std['nodes_explored']} nodes)!")

    # -----------------------------------------------------------------
    # Test 4: Contradiction Rejection Validation
    # -----------------------------------------------------------------
    print("\n--- [Test 4: Input Contradiction Rejection] ---")

    # 4a. Pre-placed queen on forbidden cell
    try:
        solve_8_queens(n=8, pre_placed=["d4"], forbidden=["d4"])
        print("ERROR: Failed to reject pre-placed queen on forbidden cell")
    except ValueError as e:
        print(f"  [PASSED] Rejection 1: {e}")

    # 4b. Pre-placed queens attacking diagonally
    try:
        solve_8_queens(n=8, pre_placed=["a1", "b2"])
        print("ERROR: Failed to reject diagonal attack")
    except ValueError as e:
        print(f"  [PASSED] Rejection 2: {e}")

    # 4c. Pre-placed queens in the same column
    try:
        solve_8_queens(n=8, pre_placed=["c2", "c7"])
        print("ERROR: Failed to reject column conflict")
    except ValueError as e:
        print(f"  [PASSED] Rejection 3: {e}")

    # -----------------------------------------------------------------
    # Test 5: Unsolvable Board Configuration
    # -----------------------------------------------------------------
    print("\n--- [Test 5: Valid Inputs with NO Possible Solutions] ---")
    # Queens at a8, b6, c4, d2 leave no non-attacking spots for row 4
    unsolvable_queens = ["a8", "b6", "c4", "d2"]
    has_sol = has_valid_arrangement(n=8, pre_placed=unsolvable_queens)
    print(f"Pre-placed queens         : {unsolvable_queens}")
    print(f"Valid arrangement exists? : {has_sol} (Correctly detected: NO solution exists)")
    assert not has_sol

    print("\n" + "=" * 72)
    print("        All Tests and Requirements Completed Successfully!        ")
    print("=" * 72)
