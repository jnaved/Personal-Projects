/*
 * High Performance Computing - Lab 4
 * Parallel Merge Sort using OpenMP Sections & Sequential Comparison
 *
 * Overview:
 *  - Merge Sort is a divide-and-conquer algorithm that recursively splits an
 *    array into two halves (left and right), sorts each half independently,
 *    and then merges the sorted halves back together.
 *  - In OpenMP, '#pragma omp sections' and '#pragma omp section' distribute
 *    independent blocks of code across threads in a team.
 *  - In this implementation, the two recursive branch calls (sorting left and right)
 *    are assigned to independent sections within an OpenMP parallel sections block.
 *  - Nested parallelism is enabled (omp_set_max_active_levels)
 *    to allow recursive parallel sections up to a configured MAX_DEPTH (depth 17).
 *  - A threshold cutoff (MERGE_CUTOFF) is utilized: when the subarray size falls below
 *    this threshold or depth limit is reached, execution switches to sequential merge sort
 *    to eliminate thread creation overhead on small partitions.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// =============================================================
// CONFIGURATION PARAMETERS
// =============================================================
#define ARRAY_SIZE    10000000  // 10 Million elements (~38.15 MB)
#define MAX_DEPTH     17        // Max recursion depth for parallel sections (up to 2^17 = 131,072 leaves)
#define MERGE_CUTOFF  50000     // Switch to sequential sort below this subarray size
#define RANDOM_SEED   42        // Fixed seed for deterministic benchmark comparison

// =============================================================
// MERGE UTILITY FUNCTION
// =============================================================
// Merges two sorted subarrays arr[left..mid] and arr[mid+1..right]
// using preallocated temporary buffer 'temp'.
void merge(int *arr, int *temp, int left, int mid, int right) {
    int i = left;      // Pointer for left subarray
    int j = mid + 1;   // Pointer for right subarray
    int k = left;      // Pointer for merged array in temp

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    // Copy sorted elements back from temp to arr
    memcpy(&arr[left], &temp[left], (right - left + 1) * sizeof(int));
}

// =============================================================
// SEQUENTIAL MERGE SORT
// =============================================================
void sequential_merge_sort(int *arr, int *temp, int left, int right) {
    if (left >= right) {
        return;
    }

    int mid = left + (right - left) / 2;
    sequential_merge_sort(arr, temp, left, mid);
    sequential_merge_sort(arr, temp, mid + 1, right);
    merge(arr, temp, left, mid, right);
}

// =============================================================
// PARALLEL MERGE SORT USING OPENMP SECTIONS
// =============================================================
// Uses '#pragma omp parallel sections' to execute left and right
// recursive sorting concurrently in two independent sections.
void parallel_merge_sort_sections(int *arr, int *temp, int left, int right, int depth) {
    if (left >= right) {
        return;
    }

    // Fall back to sequential sorting if max depth reached or size is below cutoff
    if (depth <= 0 || (right - left + 1) <= MERGE_CUTOFF) {
        sequential_merge_sort(arr, temp, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            parallel_merge_sort_sections(arr, temp, left, mid, depth - 1);
        }

        #pragma omp section
        {
            parallel_merge_sort_sections(arr, temp, mid + 1, right, depth - 1);
        }
    }

    // After both sections complete, merge the two sorted halves
    merge(arr, temp, left, mid, right);
}

// Top-level wrapper for parallel merge sort
void run_parallel_merge_sort(int *arr, int *temp, int n, int max_depth) {
    // Enable nested parallelism by configuring maximum active levels (OpenMP 3.0+)
    omp_set_max_active_levels(max_depth);

    parallel_merge_sort_sections(arr, temp, 0, n - 1, max_depth);
}

// =============================================================
// VALIDATION & HELPER FUNCTIONS
// =============================================================
// Verify if the array is sorted in non-decreasing order
int is_sorted(const int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return 0; // Not sorted
        }
    }
    return 1; // Sorted
}

// Calculate XOR checksum to ensure data integrity
long long compute_checksum(const int *arr, int n) {
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Populate array with pseudorandom integers
void generate_random_array(int *arr, int n) {
    srand(RANDOM_SEED);
    for (int i = 0; i < n; i++) {
        arr[i] = rand();
    }
}

// =============================================================
// MAIN BENCHMARK DRIVER
// =============================================================
int main(void) {
    int num_procs = omp_get_num_procs();
    int max_threads = omp_get_max_threads();

    printf("========================================================================================\n");
    printf("                  HPC LAB 4: MERGE SORT BENCHMARK (SECTIONS vs SEQUENTIAL)              \n");
    printf("========================================================================================\n");
    printf(" System Hardware & OpenMP Environment Configuration:\n");
    printf("   - Available CPU Processors : %d\n", num_procs);
    printf("   - Max OpenMP Threads       : %d\n", max_threads);
    printf("   - Max Recursion Depth      : %d (allows up to 2^%d = %d parallel leaves)\n", 
           MAX_DEPTH, MAX_DEPTH, 1 << MAX_DEPTH);
    printf("   - Sequential Cutoff Size   : %d elements\n", MERGE_CUTOFF);
    printf("   - Dataset Size             : %d elements (%.2f MB)\n", 
           ARRAY_SIZE, (double)(ARRAY_SIZE * sizeof(int)) / (1024.0 * 1024.0));
    printf("========================================================================================\n\n");

    // Allocate memory for data arrays and merge buffers
    int *arr_seq = (int *)malloc(sizeof(int) * ARRAY_SIZE);
    int *arr_par = (int *)malloc(sizeof(int) * ARRAY_SIZE);
    int *temp_seq = (int *)malloc(sizeof(int) * ARRAY_SIZE);
    int *temp_par = (int *)malloc(sizeof(int) * ARRAY_SIZE);

    if (!arr_seq || !arr_par || !temp_seq || !temp_par) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        if (arr_seq) free(arr_seq);
        if (arr_par) free(arr_par);
        if (temp_seq) free(temp_seq);
        if (temp_par) free(temp_par);
        return 1;
    }

    // Initialize arrays with identical pseudorandom values
    generate_random_array(arr_seq, ARRAY_SIZE);
    memcpy(arr_par, arr_seq, sizeof(int) * ARRAY_SIZE);

    long long initial_checksum = compute_checksum(arr_seq, ARRAY_SIZE);

    // -------------------------------------------------------------
    // 1. Sequential Merge Sort Benchmark
    // -------------------------------------------------------------
    printf("Executing Sequential Merge Sort...\n");
    double start_seq = omp_get_wtime();
    sequential_merge_sort(arr_seq, temp_seq, 0, ARRAY_SIZE - 1);
    double end_seq = omp_get_wtime();
    double time_seq = end_seq - start_seq;

    int seq_sorted = is_sorted(arr_seq, ARRAY_SIZE);
    long long seq_checksum = compute_checksum(arr_seq, ARRAY_SIZE);
    int seq_valid = seq_sorted && (seq_checksum == initial_checksum);

    // -------------------------------------------------------------
    // 2. Parallel Merge Sort using Sections Benchmark
    // -------------------------------------------------------------
    printf("Executing Parallel Merge Sort (OpenMP Sections)...\n\n");
    double start_par = omp_get_wtime();
    run_parallel_merge_sort(arr_par, temp_par, ARRAY_SIZE, MAX_DEPTH);
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;

    int par_sorted = is_sorted(arr_par, ARRAY_SIZE);
    long long par_checksum = compute_checksum(arr_par, ARRAY_SIZE);
    int par_valid = par_sorted && (par_checksum == initial_checksum);

    // Calculate speedup and efficiency
    double speedup = time_seq / time_par;
    double efficiency = (speedup / num_procs) * 100.0;

    // -------------------------------------------------------------
    // 3. Tabular Output
    // -------------------------------------------------------------
    printf("+----------------------------------+----------------+----------------+------------+----------------+\n");
    printf("| Implementation                   | Execution (s)  | Execution (ms) | Speedup    | Correctness    |\n");
    printf("+----------------------------------+----------------+----------------+------------+----------------+\n");
    printf("| Sequential Merge Sort            | %11.4f s  | %11.2f ms  | %9.2fx  | %-14s |\n", 
           time_seq, time_seq * 1000.0, 1.00, seq_valid ? "PASSED (OK)" : "FAILED");
    printf("| Parallel Merge Sort (Sections)   | %11.4f s  | %11.2f ms  | %9.2fx  | %-14s |\n", 
           time_par, time_par * 1000.0, speedup, par_valid ? "PASSED (OK)" : "FAILED");
    printf("+----------------------------------+----------------+----------------+------------+----------------+\n\n");

    printf("========================================================================================\n");
    printf(" PERFORMANCE SUMMARY & ANALYSIS\n");
    printf("========================================================================================\n");
    printf("  Sequential Execution Time : %10.4f seconds (%10.2f ms)\n", time_seq, time_seq * 1000.0);
    printf("  Parallel Execution Time   : %10.4f seconds (%10.2f ms)\n", time_par, time_par * 1000.0);
    printf("  Absolute Time Saved       : %10.4f seconds (%10.2f ms)\n", 
           (time_seq - time_par), (time_seq - time_par) * 1000.0);
    printf("  Measured Speedup (S)      : %10.2fx (T_seq / T_par)\n", speedup);
    printf("  Parallel Efficiency (E)   : %10.2f%% (Speedup / %d Cores)\n", efficiency, num_procs);
    printf("  Verification Status       : %s\n", 
           (seq_valid && par_valid) ? "SUCCESS - All elements correctly sorted with data integrity preserved." 
                                    : "FAILURE - Sorting order or checksum mismatch detected.");
    printf("========================================================================================\n");

    // Clean up allocated memory
    free(arr_seq);
    free(arr_par);
    free(temp_seq);
    free(temp_par);

    return 0;
}
