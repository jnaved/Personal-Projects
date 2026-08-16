"""
================================================================================
Employee Name Sorting and Duplicate Detection Algorithm (Problem 1)
================================================================================
Problem Statement:
Design an algorithm to arrange n employee names in lexicographical order 
and display all duplicate names without using built-in sorting functions.

COMPLEXITY ANALYSIS (ENTIRE SOLUTION):
--------------------------------------------------------------------------------
Let N = total_employee_count (number of employee names)
Let K = max_name_len (maximum length of an employee name string)

Note on Notation:
The time complexity is O(N * K * log N), NOT O(N * log(N * K)).
The log factor comes from partitioning N array elements (log2 N levels),
while the string length K acts as a linear multiplier for each comparison.

1. TIME COMPLEXITY:
   - Best-Case Time Complexity:    O(N * K * log N)
   - Average-Case Time Complexity: O(N * K * log N)
   - Worst-Case Time Complexity:   O(N * K * log N)
   - Duplicate Scan Phase:          O(N * K)

2. AUXILIARY SPACE COMPLEXITY:
   - Auxiliary Space Complexity:   O(N * K)

3. DUPLICATE DETECTION WITHOUT SORTING (HASH TABLE):
   - Best / Average / Worst-case Time:     O(N * K)
   - Auxiliary Space Complexity:           O(N * K)
================================================================================
"""

from typing import List, Dict


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


def custom_merge(left_sublist: List[str], right_sublist: List[str]) -> List[str]:
    """
    Merges two sorted lists into a single sorted list using custom string comparison.
    """
    merged_result = []
    left_index = 0
    right_index = 0
    
    while left_index < len(left_sublist) and right_index < len(right_sublist):
        comparison_result = compare_strings(
            left_sublist[left_index], 
            right_sublist[right_index]
        )
        if comparison_result <= 0:
            merged_result.append(left_sublist[left_index])
            left_index += 1
        else:
            merged_result.append(right_sublist[right_index])
            right_index += 1
            
    while left_index < len(left_sublist):
        merged_result.append(left_sublist[left_index])
        left_index += 1
        
    while right_index < len(right_sublist):
        merged_result.append(right_sublist[right_index])
        right_index += 1
        
    return merged_result


def merge_sort_names(employee_names: List[str]) -> List[str]:
    """
    Sorts a list of employee names in lexicographical order using custom Merge Sort.
    Does not use any built-in sorting functions (such as sort() or sorted()).
    """
    total_employee_count = len(employee_names)
    if total_employee_count <= 1:
        return list(employee_names)
    
    middle_index = total_employee_count // 2
    left_half = merge_sort_names(employee_names[:middle_index])
    right_half = merge_sort_names(employee_names[middle_index:])
    
    return custom_merge(left_half, right_half)


def find_duplicates_after_sorting(sorted_employee_names: List[str]) -> Dict[str, int]:
    """
    Identifies duplicate names from an already sorted list in a single linear pass.
    """
    duplicate_names_map = {}
    total_names_count = len(sorted_employee_names)
    if total_names_count <= 1:
        return duplicate_names_map

    current_index = 0
    while current_index < total_names_count:
        occurrence_count = 1
        scan_index = current_index + 1
        
        while (
            scan_index < total_names_count and 
            compare_strings(sorted_employee_names[current_index], sorted_employee_names[scan_index]) == 0
        ):
            occurrence_count += 1
            scan_index += 1
        
        if occurrence_count > 1:
            target_name = sorted_employee_names[current_index]
            duplicate_names_map[target_name] = occurrence_count
            
        current_index = scan_index

    return duplicate_names_map


def find_duplicates_without_sorting(employee_names: List[str]) -> Dict[str, int]:
    """
    Detects duplicates without sorting using a Hash Table (Frequency Map).
    """
    frequency_map: Dict[str, int] = {}
    
    for name in employee_names:
        frequency_map[name] = frequency_map.get(name, 0) + 1
        
    duplicate_names_map = {
        name: count for name, count in frequency_map.items() if count > 1
    }
            
    return duplicate_names_map


if __name__ == "__main__":
    employee_names = ["John", "alice", "Bob", "John", "Alice", "Charlie", "bob", "David"]

    sorted_employee_names = merge_sort_names(employee_names)
    duplicate_names_map = find_duplicates_after_sorting(sorted_employee_names)

    print("Original List:", employee_names)
    print("Sorted List:", sorted_employee_names)
    if duplicate_names_map:
        print("Duplicate Names:")
        for name, count in duplicate_names_map.items():
            print(f"  '{name}' occurs {count} times")
    else:
        print("No duplicate names exist.")
