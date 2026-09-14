/*
 * High Performance Computing - Lab 4
 * Parallel Quick Sort using OpenMP Tasks & Sequential Comparison
 *
 * Overview:
 *  - Quick Sort is a divide-and-conquer algorithm that selects a pivot,
 *    partitions the array such that all elements smaller than the pivot
 *    are placed to its left and all elements greater are placed to its right,
 *    and then recursively sorts the left and right subarrays.
 *  - In OpenMP, '#pragma omp task' defines an explicit unit of work that can
 *    be dynamically scheduled across any available thread in the thread team.
 *  - Tasking Pattern for Recursive Divide-and-Conquer:
 *      1. '#pragma omp parallel': Creates a pool of worker threads.
 *      2. '#pragma omp single': Ensures only one thread initiates the root call.
 *      3. '#pragma omp task': Wraps the recursive sorting of subarrays.
 *      4. '#pragma omp taskwait': Synchronizes child tasks before returning.
 *  - A threshold cutoff (TASK_CUTOFF) is utilized: when the subarray size drops
 *    below this threshold, the algorithm switches to sequential Quick Sort to
 *    avoid task scheduling and queuing overhead on small partitions.
 *  - Worker Thread Allocation (31 Threads):
 *    Allocates 31 worker threads (oversubscription rule: 2 * cores - 1).
 *    For memory-bound Quick Sort, oversubscribing workers provides latency
 *    hiding against DRAM memory stalls and mitigates load imbalance across
 *    asymmetric partitions via OpenMP dynamic work-stealing.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// =============================================================
// CONFIGURATION PARAMETERS
// =============================================================
#define ARRAY_SIZE   10000000  // 10 Million elements (~38.15 MB)
#define NUM_WORKERS  31        // 31 worker threads (Oversubscription: 2 * cores - 1)
#define TASK_CUTOFF  50000     // Switch to sequential sort below this size
#define RANDOM_SEED  42        // Fixed seed for deterministic benchmark comparison

// =============================================================
// PARTITION & HELPER FUNCTIONS
// =============================================================
// Swap two integer values
static inline void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Basic Lomuto partition: pivot is always the last element (arr[high])
int partition(int *arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

// =============================================================
// SEQUENTIAL QUICK SORT
// =============================================================
void sequential_quick_sort(int *arr, int low, int high) {
    if (low < high) {
        int p = partition(arr, low, high);
        sequential_quick_sort(arr, low, p - 1);
        sequential_quick_sort(arr, p + 1, high);
    }
}

// =============================================================
// PARALLEL QUICK SORT USING OPENMP TASKS
// =============================================================
// Spawns OpenMP tasks for independent recursive subarray partitions
void parallel_quick_sort_tasks(int *arr, int low, int high, int cutoff) {
    if (low < high) {
        // Fall back to fast sequential quicksort below threshold
        if ((high - low + 1) <= cutoff) {
            sequential_quick_sort(arr, low, high);
            return;
        }

        int p = partition(arr, low, high);

        // Spawn independent task for left partition
        #pragma omp task firstprivate(arr, low, p, cutoff)
        parallel_quick_sort_tasks(arr, low, p - 1, cutoff);

        // Spawn independent task for right partition
        #pragma omp task firstprivate(arr, high, p, cutoff)
        parallel_quick_sort_tasks(arr, p + 1, high, cutoff);

        // Wait for child tasks to complete
        #pragma omp taskwait
    }
}

// Top-level wrapper for parallel quick sort
void run_parallel_quick_sort(int *arr, int n, int num_threads, int cutoff) {
    #pragma omp parallel num_threads(num_threads)
    {
        // Single thread kicks off the initial root task; all threads execute tasks
        #pragma omp single nowait
        {
            parallel_quick_sort_tasks(arr, 0, n - 1, cutoff);
        }
    }
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

// Calculate checksum to verify data integrity
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
    int max_threads = NUM_WORKERS;

    printf("========================================================================================\n");
    printf("                  HPC LAB 4: QUICK SORT BENCHMARK (TASKS vs SEQUENTIAL)                 \n");
    printf("========================================================================================\n");
    printf(" System Hardware & OpenMP Environment Configuration:\n");
    printf("   - Available CPU Processors : %d\n", num_procs);
    printf("   - Worker Threads Allocated : %d (Oversubscribed: 2*cores - 1 for latency hiding)\n", max_threads);
    printf("   - Task Cutoff Size         : %d elements\n", TASK_CUTOFF);
    printf("   - Dataset Size             : %d elements (%.2f MB)\n", 
           ARRAY_SIZE, (double)(ARRAY_SIZE * sizeof(int)) / (1024.0 * 1024.0));
    printf("========================================================================================\n\n");

    // Allocate memory for sequential and parallel arrays
    int *arr_seq = (int *)malloc(sizeof(int) * ARRAY_SIZE);
    int *arr_par = (int *)malloc(sizeof(int) * ARRAY_SIZE);

    if (!arr_seq || !arr_par) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        if (arr_seq) free(arr_seq);
        if (arr_par) free(arr_par);
        return 1;
    }

    // Initialize arrays with identical pseudorandom values
    generate_random_array(arr_seq, ARRAY_SIZE);
    memcpy(arr_par, arr_seq, sizeof(int) * ARRAY_SIZE);

    long long initial_checksum = compute_checksum(arr_seq, ARRAY_SIZE);

    // -------------------------------------------------------------
    // 1. Sequential Quick Sort Benchmark
    // -------------------------------------------------------------
    printf("Executing Sequential Quick Sort...\n");
    double start_seq = omp_get_wtime();
    sequential_quick_sort(arr_seq, 0, ARRAY_SIZE - 1);
    double end_seq = omp_get_wtime();
    double time_seq = end_seq - start_seq;

    int seq_sorted = is_sorted(arr_seq, ARRAY_SIZE);
    long long seq_checksum = compute_checksum(arr_seq, ARRAY_SIZE);
    int seq_valid = seq_sorted && (seq_checksum == initial_checksum);

    // -------------------------------------------------------------
    // 2. Parallel Quick Sort using Tasks Benchmark
    // -------------------------------------------------------------
    printf("Executing Parallel Quick Sort (OpenMP Tasks)...\n\n");
    double start_par = omp_get_wtime();
    run_parallel_quick_sort(arr_par, ARRAY_SIZE, max_threads, TASK_CUTOFF);
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;

    int par_sorted = is_sorted(arr_par, ARRAY_SIZE);
    long long par_checksum = compute_checksum(arr_par, ARRAY_SIZE);
    int par_valid = par_sorted && (par_checksum == initial_checksum);

    // Calculate speedup and efficiency
    double speedup = time_seq / time_par;
    double efficiency = (speedup / max_threads) * 100.0;

    // -------------------------------------------------------------
    // 3. Tabular Output
    // -------------------------------------------------------------
    printf("+----------------------------------+----------------+----------------+------------+----------------+\n");
    printf("| Implementation                   | Execution (s)  | Execution (ms) | Speedup    | Correctness    |\n");
    printf("+----------------------------------+----------------+----------------+------------+----------------+\n");
    printf("| Sequential Quick Sort            | %11.4f s  | %11.2f ms  | %9.2fx  | %-14s |\n", 
           time_seq, time_seq * 1000.0, 1.00, seq_valid ? "PASSED (OK)" : "FAILED");
    printf("| Parallel Quick Sort (Tasks)      | %11.4f s  | %11.2f ms  | %9.2fx  | %-14s |\n", 
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
    printf("  Parallel Efficiency (E)   : %10.2f%% (Speedup / %d Threads)\n", efficiency, max_threads);
    printf("  Verification Status       : %s\n", 
           (seq_valid && par_valid) ? "SUCCESS - All elements correctly sorted with data integrity preserved." 
                                    : "FAILURE - Sorting order or checksum mismatch detected.");
    printf("========================================================================================\n");

    // Clean up allocated memory
    free(arr_seq);
    free(arr_par);

    return 0;
}
