"""
================================================================================
Problem 3 First Unique Employee
================================================================================
Problem Description:
A company records the sequence in which employees access a system. An employee 
is called unique if their name occurs exactly once in the complete sequence.

Given a sequence of n employee names, design two solutions to find the first 
unique name:
1. Solution 1: Uses ONLY the given list and repeated comparisons.
2. Solution 2: Uses an additional data structure (Hash Map / Frequency Map).

Example:
Input: ["Amit", "Neha", "Amit", "Rahul", "Neha", "Karan", "Rahul"]
Output: "Karan"

COMPLEXITY COMPARISON:
--------------------------------------------------------------------------------
Approach 1 (List-Only Repeated Comparisons):
- Best-Case Time:    O(n * k)   (When first element is unique and compared across sequence)
- Average-Case Time: O(n^2 * k)
- Worst-Case Time:   O(n^2 * k) (When all elements are duplicates or unique is at the end)
- Auxiliary Space:   O(1)

Approach 2 (Hash Map / Frequency Map):
- Best-Case Time:    O(n * k)
- Average-Case Time: O(n * k)
- Worst-Case Time:   O(n * k)
- Auxiliary Space:   O(n * k)
================================================================================
"""

from typing import List, Optional, Dict


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


def find_first_unique_list_only(employee_sequence: List[str]) -> Optional[str]:
    """
    Approach 1: Finds the first unique employee using ONLY the given list and 
    repeated comparisons. No additional data structures are used.
    
    Time Complexity:  O(n^2 * k) where n = len(sequence), k = max_name_len
    Auxiliary Space:  O(1)
    """
    sequence_length = len(employee_sequence)

    for current_index in range(sequence_length):
        target_name = employee_sequence[current_index]
        is_duplicate = False

        for scan_index in range(sequence_length):
            if current_index != scan_index:
                # Compare target_name with every other element in the list using custom character-by-character comparison
                if compare_strings(target_name, employee_sequence[scan_index]) == 0:
                    is_duplicate = True
                    break  # Found duplicate, move to next target_name

        if not is_duplicate:
            return target_name

    return None  # No unique employee found


def find_first_unique_hash_map(employee_sequence: List[str]) -> Optional[str]:
    """
    Approach 2: Finds the first unique employee using an additional data structure 
    (Hash Table / Frequency Map).
    
    Time Complexity:  O(n * k)
    Auxiliary Space:  O(n * k)
    """
    frequency_map: Dict[str, int] = {}

    # Pass 1: Build frequency count dictionary
    for name in employee_sequence:
        frequency_map[name] = frequency_map.get(name, 0) + 1

    # Pass 2: Find the first name in original sequence order with count == 1
    for name in employee_sequence:
        if frequency_map[name] == 1:
            return name

    return None  # No unique employee found


if __name__ == "__main__":
    test_cases = [
        {
            "description": "Standard Assignment Example",
            "sequence": ["Amit", "Neha", "Amit", "Rahul", "Neha", "Karan", "Rahul"]
        },
        {
            "description": "First Element is Unique",
            "sequence": ["UniqueUser", "Bob", "Alice", "Bob", "Alice"]
        },
        {
            "description": "No Unique Employees (All Duplicates)",
            "sequence": ["Smith", "John", "Smith", "John", "Alex", "Alex"]
        },
        {
            "description": "All Employees Unique",
            "sequence": ["Alpha", "Beta", "Gamma", "Delta"]
        },
        {
            "description": "Single Employee Sequence",
            "sequence": ["SoloEmployee"]
        }
    ]

    for test in test_cases:
        print(f"Test Case: {test['description']}")
        print(f"Sequence: {test['sequence']}")
        
        result_list = find_first_unique_list_only(test["sequence"])
        result_hash = find_first_unique_hash_map(test["sequence"])

        print(f"Approach 1 (List-Only O(n^2)):  {result_list if result_list else 'No unique employee'}")
        print(f"Approach 2 (Hash Map O(n)):    {result_hash if result_hash else 'No unique employee'}")
        
        assert result_list == result_hash, "Mismatch between approaches!"
        print()
