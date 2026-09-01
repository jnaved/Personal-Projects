'''
    Implementing Inversion Pair while sorting using Divide and Conquer
'''
import matplotlib.pyplot as plt
import random
import time
from pathlib import Path


def count_and_sort(arr: list[int], left: int, mid: int, right: int):
    left_len = mid - left + 1
    right_len = right - mid

    left_arr = arr[left: mid+1]
    right_arr = arr[mid+1:right+1]

    i, j, k = 0, 0, left
    res = 0

    while(i < left_len and j < right_len):
        # No inversion count if left_array element is less than right_array element
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
            res += (left_len - i)
        k+=1

    while i < left_len:
        arr[k] = left_arr[i]
        i+=1
        k+=1

    while j < right_len:
        arr[k] = right_arr[j]
        j+=1
        k+=1

    return res 


def count_inversions(arr: list[int], left: int, right: int):
    res = 0
    if left >= right:
        return res

    mid = left + (right - left)//2
    
    res += count_inversions(arr=arr, left=left, right=mid)
    res += count_inversions(arr=arr, left=mid+1, right=right)

    res += count_and_sort(arr=arr, left=left, mid=mid, right=right)
    return res



def inversion_pair_dac(arr: list[int]):
    n = len(arr)
    a = arr.copy()
    return count_inversions(arr=a, left=0, right=n-1)


def inversion_pair_insertion_sort(arr: list[int]):
    res = 0
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j+1] = a[j]
            res +=1
            j -= 1
        a[j+1] = key
    return res


def inversion_pair_bubble_sort(arr: list[int]):
    res = 0
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if a[j]>a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                res +=1
                swapped = True
        if not swapped:
            break
    return res


def inversion_pair(arr: list[int]):
    res = 0
    a = arr.copy()
    n = len(a)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if a[i] > a[j]:
                res += 1
    return res


def generate_random_array(count: int = 100) -> list[int]:
    random.seed(42 + count)
    array = []
    for _ in range(count):
        c = random.randint(1, count)
        array.append(c)
    return array


def run_and_plot(array_sizes: list[int]):
    dac_times = []
    iterative_times = []
    isort_times = []
    bsort_times = []

    for n in range(len(array_sizes)):
        test_arr = generate_random_array(count=array_sizes[n])

        t0 = time.perf_counter()
        inversion_pair_dac(test_arr.copy())
        dac_times.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        inversion_pair(test_arr.copy())
        iterative_times.append((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        inversion_pair_insertion_sort(test_arr.copy())
        isort_times.append((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        inversion_pair_bubble_sort(test_arr.copy())
        bsort_times.append((time.perf_counter() - t3) * 1000)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Array Size vs Execution Time (DAC & Other Sort Methods)
    ax1.plot(array_sizes, dac_times, label="DAC (Merge Sort)", color="#1f77b4", marker="o", linewidth=2)
    ax1.plot(array_sizes, isort_times, label="Insertion Sort", color="#ff7f0e", marker="s", linewidth=2)
    ax1.plot(array_sizes, bsort_times, label="Bubble Sort", color="#2ca02c", marker="^", linewidth=2)
    

    ax1.set_title("Sorting & Counting Methods Execution Time (Log Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Array Size (N) (Log Scale)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms) (Log Scale)", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Subplot 2: DAC vs Iterative Execution Time
    ax2.plot(array_sizes, dac_times, label="(DAC (Merge Sort)", color="#1f77b4", marker="o", linewidth=2.5)
    ax2.plot(array_sizes, iterative_times, label="Iterative Sort", color="#ff7f0e", marker="s", linewidth=2.5)

    ax2.set_title("DAC vs Iterative (Log - Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Array Size (N) [Log Scale]", fontsize=11)
    ax2.set_ylabel("Execution Time (ms) [Log Scale]", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.suptitle("Inversion Count: Time Complexity Analysis across Approaches", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save figure
    current_dir = Path(__file__).resolve().parent
    images_dir = current_dir.parent.parent / "images"
    save_path = images_dir / "inversion_pair_tc_analysis.png" if images_dir.exists() else current_dir / "inversion_pair_tc_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Plot Saved] Figure saved to: {save_path}")

    # Also sync with overleaf if present
    overleaf_images_dir = current_dir.parent.parent / "overleaf" / "lab_4" / "images"
    if overleaf_images_dir.exists():
        plt.savefig(overleaf_images_dir / "inversion_pair_tc_analysis.png", dpi=300)

    plt.show()



if __name__ == "__main__":
    print("Enter the number of elements :")
    elements = int(input().strip())

    print("Select Option\nUser Entered - 1\nStatic Generated - 0")
    input = int(input().strip() or 0)
    if not input:
        arr = generate_random_array(count=elements)
    else:
        print("Enter the array of integers : ")
        arr = list(map(int, input().split()))

    start_time = time.perf_counter()
    inversion_count_dac = inversion_pair_dac(arr=arr)
    time_taken = (time.perf_counter() - start_time) * 1000
    print(f"DAC Approach - Inversion Count : {inversion_count_dac}, time taken : {time_taken} ms")

    start_time = time.perf_counter()
    inversion_count = inversion_pair(arr=arr)
    time_taken = (time.perf_counter() - start_time) * 1000
    print(f"Normal Approach : Inversion Count  {inversion_count}, time taken : {time_taken} ms")
    
    run_and_plot(array_sizes=[100, 1000, 10000])