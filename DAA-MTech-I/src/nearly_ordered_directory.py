"""
================================================================================
Problem 5 Nearly Ordered Employee Directory
================================================================================
Problem Description:
An employee directory containing n names is expected to be in lexicographical 
order, but due to a data-entry error, some names may have been placed incorrectly.

Given the list:
1. Determine whether it is already sorted.
2. Identify all positions where the ordering is violated (i.e. names[i] > names[i+1]).
3. Determine whether exchanging ONLY TWO names can make the complete list sorted. 
   If possible, report the two positions (indices).

Constraint: Do NOT sort the complete array merely to determine ordering.

COMPLEXITY ANALYSIS:
--------------------------------------------------------------------------------
1. Time Complexity:
   - Best-Case Time Complexity:    O(n * k)
   - Worst-Case Time Complexity:   O(n * k)
     Scanning the array to detect inversions takes O(n * k) time. 
     Simulating the swap and verifying order takes O(n * k) time.
     Total Time = O(n * k), avoiding full array sorting O(n log n * k).

2. Auxiliary Space Complexity:     O(1)
   Only scalar index variables (inversion indices, swap candidate positions) 
   are maintained.
================================================================================
"""

from typing import List, Tuple, Optional


def compare_strings(first_name: str, second_name: str) -> int:
    """
    Compares two strings lexicographically using standard case-sensitive ASCII values.
    Returns -1 if first_name < second_name, 0 if equal, 1 if first_name > second_name.
    """
    first_string_len = len(first_name)
    second_string_len = len(second_name)
    min_len = min(first_string_len, second_string_len)

    for character_index in range(min_len):
        first_character = first_name[character_index]
        second_character = second_name[character_index]
        
        if first_character != second_character:
            return -1 if first_character < second_character else 1

    if first_string_len < second_string_len:
        return -1
    elif first_string_len > second_string_len:
        return 1
    return 0


def analyze_nearly_ordered_directory(
    names: List[str]
) -> Tuple[bool, List[int], bool, Optional[Tuple[int, int]]]:
    """
    Analyzes an employee directory to check if it is sorted, finds ordering 
    violations, and checks if a single swap can fix the list.
    
    Returns:
        (is_sorted, violation_indices, can_fix_with_single_swap, swap_positions)
    """
    total_names_count = len(names)
    if total_names_count <= 1:
        return True, [], True, None

    # Step 1: Identify all inversion indices where names[i] > names[i + 1]
    violation_indices = []
    for i in range(total_names_count - 1):
        if compare_strings(names[i], names[i + 1]) > 0:
            violation_indices.append(i)

    # If no violations exist, the list is already sorted
    if len(violation_indices) == 0:
        return True, [], True, None

    # Step 2: Identify candidate swap positions (pos_1, pos_2)
    # Case A: Exactly 1 inversion (adjacent elements swapped)
    if len(violation_indices) == 1:
        pos_1 = violation_indices[0]
        pos_2 = violation_indices[0] + 1
    # Case B: Exactly 2 inversions (non-adjacent elements swapped)
    elif len(violation_indices) == 2:
        pos_1 = violation_indices[0]
        pos_2 = violation_indices[1] + 1
    else:
        # More than 2 inversions means a single swap cannot restore ordering
        return False, violation_indices, False, None

    # Step 3: Simulate single swap and verify if array becomes sorted
    candidate_names = list(names)
    candidate_names[pos_1], candidate_names[pos_2] = candidate_names[pos_2], candidate_names[pos_1]

    # Check if candidate_names is now strictly sorted
    is_restored = True
    for i in range(total_names_count - 1):
        if compare_strings(candidate_names[i], candidate_names[i + 1]) > 0:
            is_restored = False
            break

    if is_restored:
        return False, violation_indices, True, (pos_1, pos_2)
    else:
        return False, violation_indices, False, None


if __name__ == "__main__":
    test_cases = [
        {
            "description": "Already Sorted Directory",
            "names": ["Amit", "Karan", "Neha", "Priya", "Rahul"]
        },
        {
            "description": "Single Non-Adjacent Violation (Rahul and Priya placed incorrectly)",
            "names": ["Amit", "Karan", "Neha", "Rahul", "Priya"]
        },
        {
            "description": "Single Violation (Karan and Neha swapped)",
            "names": ["Amit", "Neha", "Karan", "Rahul", "Priya"]
        },
        {
            "description": "Multiple Violations (Cannot be fixed by 1 swap)",
            "names": ["Rahul", "Priya", "Neha", "Karan", "Amit"]
        }
    ]

    for test in test_cases:
        print(f"Test Case: {test['description']}")
        print(f"Names List: {test['names']}")
        
        is_sorted, violations, can_fix, swap_pos = analyze_nearly_ordered_directory(test["names"])
        
        if is_sorted:
            print("Status: Directory is ALREADY SORTED.")
        else:
            print("Status: Directory CONTAINS ORDERING VIOLATION(S).")
            print(f"  - Violation Indices (i where names[i] > names[i+1]): {violations}")
            print(f"  - Violation Pairs:")
            for v_idx in violations:
                print(f"     * Index {v_idx} ('{test['names'][v_idx]}') > Index {v_idx+1} ('{test['names'][v_idx+1]}')")
                
            if can_fix and swap_pos:
                print(f"  - Single Swap Possible: YES")
                print(f"  - Exchanging positions {swap_pos[0]} ('{test['names'][swap_pos[0]]}') and {swap_pos[1]} ('{test['names'][swap_pos[1]]}') restores sorted order.")
            else:
                print(f"  - Single Swap Possible: NO (Multiple swaps required).")
        print()
