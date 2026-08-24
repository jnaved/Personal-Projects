"""
================================================================================
Problem 4 Stable Matrix
================================================================================
Problem Description:
An n x n integer matrix is called stable if every non-boundary (interior) element 
satisfies:
    A[i][j] = A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]

Determine whether the matrix is stable. If it is not, report:
1. Total number of violating elements.
2. Positions (i, j) of all violating elements.
3. Position having the maximum absolute difference D(i, j):
    D(i, j) = |A[i][j] - (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])|

COMPLEXITY ANALYSIS:
--------------------------------------------------------------------------------
1. Time Complexity:
   - Best-Case Time Complexity:    O(n^2)
   - Worst-Case Time Complexity:   O(n^2)
     To check stability across all non-boundary elements, the algorithm must 
     evaluate (n-2) x (n-2) interior cells. Each cell requires 4 neighbour 
     additions and 1 subtraction taking O(1) time. 
     Total Time = (n-2)^2 * O(1) = O(n^2).

2. Auxiliary Space Complexity:     O(1)  (or O(V) to store violating positions)
   - Additional Matrix Required: NO. An additional matrix is NOT required because 
     all neighbour checks read directly from the input matrix without modifying it.
================================================================================
"""

from typing import List, Tuple, Optional


def check_stable_matrix(
    matrix: List[List[int]]
) -> Tuple[bool, int, List[Tuple[int, int]], Optional[Tuple[int, int]], int]:
    """
    Checks stability of an n x n integer matrix across non-boundary elements.
    
    Returns:
        (is_stable, violation_count, violating_positions, max_diff_position, max_abs_diff)
    """
    matrix_size = len(matrix)
    
    # Matrices of size less than 3 x 3 have no non-boundary (interior) elements
    if matrix_size < 3:
        return True, 0, [], None, 0

    violating_positions = []
    max_abs_diff = -1
    max_diff_position = None

    # Check only non-boundary (interior) elements: 1 <= row_index, col_index <= matrix_size - 2
    for row_index in range(1, matrix_size - 1):
        for col_index in range(1, matrix_size - 1):
            neighbour_sum = (
                matrix[row_index - 1][col_index]      # Up
                + matrix[row_index + 1][col_index]    # Down
                + matrix[row_index][col_index - 1]    # Left
                + matrix[row_index][col_index + 1]    # Right
            )
            current_value = matrix[row_index][col_index]

            if current_value != neighbour_sum:
                violating_positions.append((row_index, col_index))
                
                # Calculate D(i, j) = |A[i][j] - Neighbour_Sum|
                abs_difference = abs(current_value - neighbour_sum)
                
                if abs_difference > max_abs_diff:
                    max_abs_diff = abs_difference
                    max_diff_position = (row_index, col_index)

    is_stable = len(violating_positions) == 0
    return is_stable, len(violating_positions), violating_positions, max_diff_position, max_abs_diff if not is_stable else 0


if __name__ == "__main__":
    test_matrices = [
        {
            "description": "Stable Matrix (All zeroes)",
            "matrix": [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        },
        {
            "description": "Unstable Matrix with Multiple Violations",
            "matrix": [
                [1, 2, 3, 4],
                [5, 100, 20, 6],
                [7, 30, 200, 8],
                [9, 10, 11, 12]
            ]
        },
        {
            "description": "Small Matrix (2x2 - No interior elements)",
            "matrix": [
                [1, 2],
                [3, 4]
            ]
        }
    ]

    for test in test_matrices:
        print(f"Test: {test['description']}")
        print("Matrix:")
        for row in test["matrix"]:
            print(" ", row)
            
        is_stable, count, positions, max_pos, max_diff = check_stable_matrix(test["matrix"])
        
        if is_stable:
            print("Result: Matrix is STABLE.")
        else:
            print("Result: Matrix is NOT STABLE.")
            print(f"  - Total Violating Elements Count: {count}")
            print(f"  - Violating Positions (row, col): {positions}")
            print(f"  - Position with Max Absolute Difference D(i, j): {max_pos}")
            print(f"  - Maximum Absolute Difference D(i, j): {max_diff}")
        print()
