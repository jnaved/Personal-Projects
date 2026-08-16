"""
================================================================================
Problem 2 – Locally Balanced Matrix
================================================================================
Problem Description:
An n x n integer matrix is locally balanced if each element A[i][j] equals the sum 
of its valid 4-directional neighbouring elements (Up, Down, Left, Right). 
Only neighbours within matrix boundaries are considered.

If the matrix is not balanced, report the first violating element in row-major 
order along with its position (i, j), value A[i][j], and neighbour sum.

COMPLEXITY ANALYSIS:
--------------------------------------------------------------------------------
1. Time Complexity:
   - Best-Case Time Complexity:    O(1)
     Occurs when the first element A[0][0] violates the condition.
     
   - Average-Case Time Complexity: O(n^2)
     On average, inspecting elements in row-major order up to a violation takes 
     O(n^2) operations.
     
   - Worst-Case Time Complexity:   O(n^2)
     Occurs when the matrix is locally balanced or the violation is at A[n-1][n-1]. 
     For an n x n matrix, there are n^2 elements, and checking valid neighbours 
     takes at most 4 operations per element (constant time O(1)). 
     Total Time = n^2 * O(1) = O(n^2).

2. Auxiliary Space Complexity:     O(1)
   Only a few primitive scalar variables (indices, neighbour sums) are used. 
   No extra matrix or data structure is allocated.
================================================================================
"""

from typing import List, Tuple, Optional


def check_locally_balanced_matrix(
    matrix: List[List[int]]
) -> Tuple[bool, Optional[Tuple[int, int]], Optional[int], Optional[int]]:
    """
    Checks whether an n x n integer matrix is locally balanced.
    
    Returns:
        A tuple (is_balanced, position, element_value, neighbour_sum):
        - is_balanced: True if all elements satisfy the condition, False otherwise.
        - position: (row, col) tuple of the first violating element, or None if balanced.
        - element_value: The value at the violating position, or None if balanced.
        - neighbour_sum: The calculated sum of valid neighbours, or None if balanced.
    """
    matrix_size = len(matrix)
    if matrix_size == 0:
        return True, None, None, None

    # 4-directional neighbour offsets: Up, Down, Left, Right
    neighbour_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Row-major traversal to find the first violation
    for row_index in range(matrix_size):
        for col_index in range(matrix_size):
            neighbour_sum = 0

            # Sum only valid neighbours within matrix boundaries
            for delta_row, delta_col in neighbour_directions:
                neighbor_row = row_index + delta_row
                neighbor_col = col_index + delta_col

                # Boundary Check
                if 0 <= neighbor_row < matrix_size and 0 <= neighbor_col < matrix_size:
                    neighbour_sum += matrix[neighbor_row][neighbor_col]

            current_element_value = matrix[row_index][col_index]

            # Violation Check
            if current_element_value != neighbour_sum:
                return (
                    False,
                    (row_index, col_index),
                    current_element_value,
                    neighbour_sum,
                )

    return True, None, None, None


if __name__ == "__main__":
    test_matrices = [
        {
            "description": "Locally Balanced Matrix (All zeroes)",
            "matrix": [
                [0, 0],
                [0, 0]
            ]
        },
        {
            "description": "Unbalanced Matrix (Violation at interior)",
            "matrix": [
                [5, 10, 5],
                [10, 99, 10],
                [5, 10, 5]
            ]
        },
        {
            "description": "Unbalanced Matrix (Violation at corner A[0][0])",
            "matrix": [
                [10, 2],
                [3, 4]
            ]
        }
    ]

    for test in test_matrices:
        print(f"Test: {test['description']}")
        print("Matrix:")
        for row in test["matrix"]:
            print(" ", row)
            
        is_balanced, pos, val, n_sum = check_locally_balanced_matrix(test["matrix"])
        
        if is_balanced:
            print("Result: Matrix is LOCALLY BALANCED.")
        else:
            print("Result: Matrix is NOT LOCALLY BALANCED.")
            print("  First Violating Element in Row-Major Order:")
            print(f"   - Position: (row={pos[0]}, col={pos[1]})")
            print(f"   - Value: A[{pos[0]}][{pos[1]}] = {val}")
            print(f"   - Sum of Valid Neighbours: {n_sum}")
        print()
