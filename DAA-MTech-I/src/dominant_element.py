"""
================================================================================
Problem 6 Dominant Element
================================================================================
Problem Description:
Given an integer array of size n, determine whether there exists an element that 
occurs more than half of the time (> n / 2).

Example:
Input: [2, 2, 1, 2, 3, 2, 2] (n = 7)
Dominant Element: 2 (Occurs 5 times, 5 > 3.5)

Develop two different solutions:
1. Solution 1: Repeated Counting / Comparison (Brute Force).
2. Solution 2: Boyer-Moore Majority Vote Algorithm (Optimal - reduces repeated work).

COMPLEXITY COMPARISON:
--------------------------------------------------------------------------------
Solution 1 (Repeated Counting / Brute Force):
- Best-Case Time:    O(n)   (When first element is dominant)
- Average-Case Time: O(n^2)
- Worst-Case Time:   O(n^2)
- Auxiliary Space:   O(1)

Solution 2 (Boyer-Moore Majority Vote Algorithm):
- Best-Case Time:    O(n)
- Average-Case Time: O(n)
- Worst-Case Time:   O(n)
- Auxiliary Space:   O(1)  (No additional storage required!)
================================================================================
"""

from typing import List, Tuple, Optional


def find_dominant_element_repeated_counting(numbers: List[int]) -> Tuple[bool, Optional[int], int]:
    """
    Solution 1: Repeated counting / brute force.
    For each element, counts its frequency across the entire array.
    
    Time Complexity:  O(n^2)
    Auxiliary Space:  O(1)
    """
    array_size = len(numbers)
    if array_size == 0:
        return False, None, 0

    majority_threshold = array_size // 2

    for current_index in range(array_size):
        candidate_val = numbers[current_index]
        occurrence_count = 0

        # Count frequency of candidate_val across the entire array
        for scan_index in range(array_size):
            if numbers[scan_index] == candidate_val:
                occurrence_count += 1

        # Check if candidate_val occurs more than n / 2 times
        if occurrence_count > majority_threshold:
            return True, candidate_val, occurrence_count

    return False, None, 0


def find_dominant_element_boyer_moore(numbers: List[int]) -> Tuple[bool, Optional[int], int]:
    """
    Solution 2: Boyer-Moore Majority Vote Algorithm.
    Finds dominant element in linear O(n) time and O(1) auxiliary space without 
    using additional storage.
    
    Time Complexity:  O(n)
    Auxiliary Space:  O(1)
    """
    array_size = len(numbers)
    if array_size == 0:
        return False, None, 0

    # Phase 1: Find Majority Candidate
    candidate = None
    vote_counter = 0

    for num in numbers:
        if vote_counter == 0:
            candidate = num
            vote_counter = 1
        elif num == candidate:
            vote_counter += 1
        else:
            vote_counter -= 1

    # Phase 2: Verify Candidate
    actual_occurrence_count = 0
    for num in numbers:
        if num == candidate:
            actual_occurrence_count += 1

    majority_threshold = array_size // 2
    if actual_occurrence_count > majority_threshold:
        return True, candidate, actual_occurrence_count

    return False, None, 0


if __name__ == "__main__":
    test_cases = [
        {
            "description": "Standard Assignment Example",
            "numbers": [2, 2, 1, 2, 3, 2, 2]
        },
        {
            "description": "No Dominant Element",
            "numbers": [1, 2, 3, 4, 5, 6]
        },
        {
            "description": "Dominant Element at Boundary (> n/2 exact)",
            "numbers": [7, 7, 7, 1, 2]
        },
        {
            "description": "Single Element Array",
            "numbers": [42]
        },
        {
            "description": "All Elements Identical",
            "numbers": [9, 9, 9, 9, 9]
        }
    ]

    for test in test_cases:
        print(f"Test Case: {test['description']}")
        print(f"Array: {test['numbers']}")
        
        has_dom_1, val_1, count_1 = find_dominant_element_repeated_counting(test["numbers"])
        has_dom_2, val_2, count_2 = find_dominant_element_boyer_moore(test["numbers"])

        if has_dom_1:
            print(f"Solution 1 (Brute Force O(n^2)): Dominant Element = {val_1} (Occurs {count_1} times)")
        else:
            print(f"Solution 1 (Brute Force O(n^2)): No dominant element exists")

        if has_dom_2:
            print(f"Solution 2 (Boyer-Moore O(n)):   Dominant Element = {val_2} (Occurs {count_2} times)")
        else:
            print(f"Solution 2 (Boyer-Moore O(n)):   No dominant element exists")
            
        assert has_dom_1 == has_dom_2 and val_1 == val_2, "Mismatch between solutions!"
        print()
