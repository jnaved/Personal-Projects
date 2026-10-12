/**
 * =====================================================================================
 * Design and Analysis of Algorithms (CSDS103) - Lab 5, Problem 2
 * =====================================================================================
 * Problem Statement:
 *   1. Create a dataset consisting of a vector of at least 1,000,000 positive integers 
 *      using the POSIX random number generator functions srand48(), lrand48(), and/or 
 *      nrand48().
 *   2. Write a function to detect and print the number of duplicates found in the input.
 *   3. Write another function to round off each duplicate to the nearest free integral 
 *      value (> 0), ensuring the final vector contains only unique positive integers.
 *   4. Measure the time required to remove/resolve the duplicates.
 *   5. Discuss various approaches to time programs in Linux and employ the approach with 
 *      the finest resolution.
 *   6. Assignment 3 Extension: Output the time in microseconds (µs) and devise the optimal 
 *      approach to achieve high-fidelity microsecond resolution.
 * =====================================================================================
 * 
 * THEORETICAL DISCUSSION: APPROACHES TO TIMING PROGRAMS IN LINUX
 * -------------------------------------------------------------------------------------
 * Measuring execution time in software and systems programming involves multiple layers 
 * of abstractions with varying resolutions, costs, and clock properties:
 * 
 * 1. Standard C `clock()` (<time.h>):
 *    - Metric: CPU processor time consumed by the calling process (user + system).
 *    - Units: Clock ticks scaled by `CLOCKS_PER_SEC` (defined as 1,000,000 on POSIX/glibc).
 *    - Real Resolution: Coarse; typically updated by the kernel timer interrupt (~1 ms to 10 ms).
 *    - Limitations:
 *      * Does not measure real wall-clock elapsed time (e.g., ignores I/O waits and sleeping).
 *      * On multi-threaded programs, ticks accumulate across all CPU cores.
 *      * On 32-bit systems, `clock_t` wraps around after approx. 72 minutes.
 * 
 * 2. POSIX `gettimeofday()` (<sys/time.h>):
 *    - Metric: Wall-clock calendar time (seconds and microseconds since the Unix epoch).
 *    - Resolution: Microsecond (1 µs = 10^-6 seconds).
 *    - Limitations:
 *      * Non-monotonic: Subject to discontinuous jumps if system administrator changes time,
 *        or if NTP (Network Time Protocol) adjusts the clock.
 *      * Marked as obsolete in POSIX.1-2008 in favor of `clock_gettime()`.
 * 
 * 3. POSIX `clock_gettime()` (<time.h>) [FINEST TIME RESOLUTION]:
 *    - Metric: High-precision timestamp via dedicated kernel clock IDs:
 *      * `CLOCK_MONOTONIC`: Monotonically increasing clock from an arbitrary point (system boot).
 *        Cannot jump backwards; ideal for measuring elapsed duration.
 *      * `CLOCK_PROCESS_CPUTIME_ID`: High-resolution per-process CPU execution time.
 *    - Resolution: Nanosecond (1 ns = 10^-9 seconds). Hardware timer via VDSO in Linux.
 *    - Precision Check: Can query kernel clock granularity directly using `clock_getres()`.
 *    - Verdict: This provides the finest resolution available in Linux user space without
 *      falling back to non-portable assembly instructions.
 * 
 * 4. Modern C++ `std::chrono` (<chrono>):
 *    - Metric: Type-safe standard library wrapper.
 *    - Clocks: `std::chrono::steady_clock` (guaranteed monotonic) and
 *      `std::chrono::high_resolution_clock` (alias to steady_clock in GNU libstdc++ on Linux).
 *    - Resolution: Nanosecond resolution (wraps `clock_gettime(CLOCK_MONOTONIC)` internally).
 * 
 * 5. Linux Shell & System Profiling Utilities:
 *    - `/usr/bin/time -v ./executable`:
 *      * Runs external to the binary; reports wall clock, CPU percentage, user/sys time,
 *        maximum resident set size (peak RAM), minor/major page faults, and context switches.
 *    - `perf` (Linux Kernel Profiling Subsystem):
 *      * Measures hardware performance counters (CPU cycles, L1/L3 cache misses, branch mispredictions).
 *      * Command: `perf stat ./executable`
 * 
 * =====================================================================================
 */

#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <ctime>       // clock(), clock_gettime(), clock_getres()
#include <sys/time.h>  // gettimeofday()
#include <chrono>      // std::chrono

// =====================================================================================
// STEP 1: DATASET GENERATION USING POSIX srand48() AND lrand48()
// =====================================================================================

/**
 * @brief Generates a vector of positive integers using POSIX 48-bit LCG functions.
 * 
 * @param n Number of elements to generate (minimum 1,000,000 as per specification).
 * @param max_val Upper bound for the values generated: range is [1, max_val].
 *                Setting max_val close to n guarantees realistic duplicate density.
 * @param seed Random seed for reproducible pseudo-random sequences.
 * @return std::vector<long> Vector of strictly positive integers (> 0).
 */
std::vector<long> generate_dataset(size_t n, long max_val = 1500000, long seed = 42) {
    std::cout << "\n[Step 1: Dataset Generation]" << std::endl;
    std::cout << "  Target Size (N)     : " << n << " entries" << std::endl;
    std::cout << "  Value Range         : [1, " << max_val << "]" << std::endl;
    std::cout << "  RNG Functions Used  : srand48(" << seed << ") and lrand48()" << std::endl;

    // Seed the 48-bit pseudo-random number generator
    srand48(seed);

    std::vector<long> dataset;
    dataset.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        // lrand48() returns non-negative 31-bit integers in [0, 2^31 - 1].
        // Modulo max_val and add 1 to ensure strictly positive integers in [1, max_val].
        long val = (lrand48() % max_val) + 1;
        dataset.push_back(val);
    }

    std::cout << "  Status              : Successfully generated " << dataset.size() 
              << " positive integers." << std::endl;
    return dataset;
}

// =====================================================================================
// STEP 2: DUPLICATE DETECTION AND ANALYSIS
// =====================================================================================

/**
 * @brief Analyzes and prints statistics regarding duplicate values in the vector.
 * 
 * @param vec The dataset to inspect.
 * @return size_t Total number of redundant duplicate instances.
 */
size_t print_duplicates(const std::vector<long>& vec) {
    std::unordered_map<long, size_t> freq_map;
    freq_map.reserve(vec.size());

    for (long val : vec) {
        freq_map[val]++;
    }

    size_t unique_values = freq_map.size();
    size_t duplicate_occurrences = 0;
    size_t values_with_duplicates = 0;

    // Track the highest occurrence count for reporting
    long most_frequent_val = -1;
    size_t max_frequency = 0;

    for (const auto& [val, count] : freq_map) {
        if (count > 1) {
            values_with_duplicates++;
            duplicate_occurrences += (count - 1);
            if (count > max_frequency) {
                max_frequency = count;
                most_frequent_val = val;
            }
        }
    }

    std::cout << "\n----------------------------------------------------------------------" << std::endl;
    std::cout << "                      DUPLICATE AUDIT REPORT                          " << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "  Total Vector Elements        : " << vec.size() << std::endl;
    std::cout << "  Unique Distinct Values       : " << unique_values << std::endl;
    std::cout << "  Distinct Values w/ Duplicates: " << values_with_duplicates << std::endl;
    std::cout << "  Total Redundant Duplicates   : " << duplicate_occurrences << " ("
              << std::fixed << std::setprecision(2)
              << (double(duplicate_occurrences) / double(vec.size()) * 100.0) << "% of dataset)" << std::endl;

    if (values_with_duplicates > 0) {
        std::cout << "  Highest Multiplicity Element : Value " << most_frequent_val 
                  << " (appeared " << max_frequency << " times)" << std::endl;
    } else {
        std::cout << "  Integrity Check              : All entries are strictly unique (0 duplicates)." << std::endl;
    }
    std::cout << "----------------------------------------------------------------------" << std::endl;

    return duplicate_occurrences;
}

// =====================================================================================
// STEP 3: ROUND OFF DUPLICATES TO NEAREST FREE INTEGRAL VALUE
// =====================================================================================

/**
 * @brief Resolves duplicates by rounding each duplicate to the nearest free integral
 *        value (> 0) that is not currently occupied in the dataset.
 * 
 * Algorithmic Strategy:
 *   1. Populate a hash set of initially encountered occupied values.
 *   2. For each element that is a duplicate of an already occupied value:
 *      Search candidate offsets delta = 1, 2, 3, ...
 *      Test (val - delta) if > 0, then (val + delta).
 *      Pick the nearest candidate not present in the set, mark it occupied,
 *      and overwrite vec[i] with the new unique integer.
 * 
 * @param vec The dataset to deduplicate in-place.
 * @return size_t Count of duplicates that were successfully relocated.
 */
size_t resolve_duplicates_nearest_free(std::vector<long>& vec) {
    std::unordered_set<long> occupied;
    occupied.reserve(vec.size() * 2);

    size_t resolved_count = 0;

    for (size_t i = 0; i < vec.size(); ++i) {
        long current = vec[i];

        // If this value is not yet in our occupied set, claim it
        if (occupied.find(current) == occupied.end()) {
            occupied.insert(current);
        } else {
            // Value is a duplicate: search for the nearest free positive integer
            resolved_count++;
            long delta = 1;
            long candidate = -1;

            while (true) {
                // Check candidate below: current - delta (must be positive > 0)
                long lower = current - delta;
                if (lower > 0 && occupied.find(lower) == occupied.end()) {
                    candidate = lower;
                    break;
                }

                // Check candidate above: current + delta
                long upper = current + delta;
                if (occupied.find(upper) == occupied.end()) {
                    candidate = upper;
                    break;
                }

                delta++;
            }

            // Claim the new nearest free integral value
            occupied.insert(candidate);
            vec[i] = candidate;
        }
    }

    return resolved_count;
}

// =====================================================================================
// STEP 4 & 5: BENCHMARK WITH MULTIPLE TIMERS & FINEST RESOLUTION ANALYSIS
// =====================================================================================

/**
 * @brief Demonstrates duplicate removal and measures duration across 4 timing mechanisms,
 *        highlighting POSIX clock_gettime() for nanosecond precision.
 */
void execute_timed_deduplication(std::vector<long>& dataset) {
    std::cout << "\n[Step 3 & 4: Executing Timed Duplicate Resolution]" << std::endl;
    std::cout << "  Resolving duplicates to nearest free positive integers..." << std::endl;

    // --- Query Linux Clock Resolution via clock_getres ---
    struct timespec clock_res;
    if (clock_getres(CLOCK_MONOTONIC, &clock_res) == 0) {
        std::cout << "  [System Info] CLOCK_MONOTONIC reported resolution: "
                  << clock_res.tv_sec << "s " << clock_res.tv_nsec << "ns" << std::endl;
    }

    // Prepare timing variables across all approaches
    // 1. clock()
    clock_t c_start = clock();

    // 2. gettimeofday()
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, nullptr);

    // 3. clock_gettime(CLOCK_MONOTONIC) - Finest POSIX resolution
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // 4. std::chrono::high_resolution_clock
    auto chrono_start = std::chrono::high_resolution_clock::now();

    // --- EXECUTE THE CORE OPERATION ---
    size_t duplicates_resolved = resolve_duplicates_nearest_free(dataset);

    // Stop all timers immediately
    auto chrono_end = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    gettimeofday(&tv_end, nullptr);
    clock_t c_end = clock();

    // Compute durations in both Microseconds (us) and Milliseconds (ms)
    // 1. clock() [POSIX CLOCKS_PER_SEC = 1,000,000; ticks directly represent microseconds]
    double clock_us = (double(c_end - c_start) * 1000000.0) / double(CLOCKS_PER_SEC);
    double clock_ms = clock_us / 1000.0;

    // 2. gettimeofday() [POSIX native microsecond struct timeval: tv_sec & tv_usec]
    long tv_usec = (tv_end.tv_sec - tv_start.tv_sec) * 1000000L + (tv_end.tv_usec - tv_start.tv_usec);
    double tv_us = double(tv_usec);
    double tv_ms = tv_us / 1000.0;

    // 3. clock_gettime(CLOCK_MONOTONIC) [Nanoseconds -> fractional microseconds & ms]
    long long ts_nsec = (long long)(ts_end.tv_sec - ts_start.tv_sec) * 1000000000LL + 
                        (long long)(ts_end.tv_nsec - ts_start.tv_nsec);
    double ts_us = double(ts_nsec) / 1000.0;
    double ts_ms = double(ts_nsec) / 1000000.0;

    // 4. std::chrono [High-resolution chrono duration in nanoseconds & microseconds]
    auto chrono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(chrono_end - chrono_start).count();
    double chrono_us = double(chrono_ns) / 1000.0;
    double chrono_ms = double(chrono_ns) / 1000000.0;

    // Display comparison table of all timing utilities with microseconds and milliseconds
    std::cout << "\n================================================================================================" << std::endl;
    std::cout << "            LINUX TIMING UTILITIES COMPARISON (MICROSECONDS [µs] & MILLISECONDS [ms])           " << std::endl;
    std::cout << "================================================================================================" << std::endl;
    std::cout << std::left << std::setw(32) << "Timing Approach / Function"
              << std::setw(18) << "Reported Value"
              << std::setw(18) << "Time (us)"
              << std::setw(16) << "Time (ms)"
              << std::setw(16) << "Theoretical Res." << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

    std::cout << std::left << std::setw(32) << "1. clock() [<ctime>]"
              << std::setw(18) << (std::to_string(c_end - c_start) + " ticks")
              << std::fixed << std::setprecision(2) << std::setw(18) << clock_us
              << std::fixed << std::setprecision(4) << std::setw(16) << clock_ms
              << std::setw(16) << "~1 - 10 ms" << std::endl;

    std::cout << std::left << std::setw(32) << "2. gettimeofday() [<sys/time.h>]"
              << std::setw(18) << (std::to_string(tv_usec) + " us")
              << std::fixed << std::setprecision(2) << std::setw(18) << tv_us
              << std::fixed << std::setprecision(4) << std::setw(16) << tv_ms
              << std::setw(16) << "1 us" << std::endl;

    std::cout << std::left << std::setw(32) << "3. clock_gettime() [MONOTONIC]*"
              << std::setw(18) << (std::to_string(ts_nsec) + " ns")
              << std::fixed << std::setprecision(2) << std::setw(18) << ts_us
              << std::fixed << std::setprecision(4) << std::setw(16) << ts_ms
              << std::setw(16) << "1 ns (FINEST)" << std::endl;

    std::cout << std::left << std::setw(32) << "4. std::chrono::high_resolution"
              << std::setw(18) << (std::to_string(chrono_ns) + " ns")
              << std::fixed << std::setprecision(2) << std::setw(18) << chrono_us
              << std::fixed << std::setprecision(4) << std::setw(16) << chrono_ms
              << std::setw(16) << "1 ns" << std::endl;

    std::cout << "================================================================================================" << std::endl;
    std::cout << "* clock_gettime(CLOCK_MONOTONIC) is the recommended POSIX mechanism: provides true sub-microsecond" << std::endl;
    std::cout << "  resolution (1 nanosecond), eliminating integer quantization truncation.\n" << std::endl;

    // --- Explicit Assignment 3 Microsecond Output Block ---
    std::cout << "[Assignment 3 Requirement: Elapsed Time in Microseconds]" << std::endl;
    std::cout << "  Primary Measurement (clock_gettime) : " 
              << std::fixed << std::setprecision(3) << ts_us << " us (" 
              << std::setprecision(4) << ts_ms << " ms)" << std::endl;
    std::cout << "  C++11 Chrono (std::chrono)          : " 
              << std::fixed << std::setprecision(3) << chrono_us << " us" << std::endl;
    std::cout << "  POSIX Native (gettimeofday)         : " 
              << std::fixed << std::setprecision(2) << tv_us << " us" << std::endl;
    std::cout << "  Duplicates relocated to nearest free: " << duplicates_resolved << std::endl;
}

// =====================================================================================
// MAIN DRIVER
// =====================================================================================

int main(int argc, char* argv[]) {
    std::cout << "=====================================================================================" << std::endl;
    std::cout << "      DAA LAB 5 - PROBLEM 2: 1M VECTOR DATASET, DUPLICATE RESOLUTION & LINUX TIMING  " << std::endl;
    std::cout << "=====================================================================================" << std::endl;

    // Default vector size is 1,000,000 as specified in the problem statement
    size_t n = 1000000;
    long max_val = 1500000;

    if (argc > 1) {
        n = std::stoull(argv[1]);
        max_val = (argc > 2) ? std::stol(argv[2]) : (long)(n * 1.5);
    }

    // Step 1: Generate dataset
    std::vector<long> dataset = generate_dataset(n, max_val, 42);

    // Step 2: Audit duplicates prior to resolution
    std::cout << "\n[Step 2: Initial Duplicate Audit]" << std::endl;
    print_duplicates(dataset);

    // Step 3, 4, 5: Execute deduplication with multi-timer benchmarking
    execute_timed_deduplication(dataset);

    // Step 6: Post-deduplication audit to verify 100% uniqueness
    std::cout << "\n[Verification: Post-Deduplication Audit]" << std::endl;
    size_t remaining_dups = print_duplicates(dataset);

    if (remaining_dups == 0) {
        std::cout << "\n[SUCCESS] Verification passed: Dataset contains exactly " 
                  << dataset.size() << " unique positive integers.\n" << std::endl;
    } else {
        std::cerr << "\n[ERROR] Verification failed: " << remaining_dups 
                  << " duplicate entries still remain.\n" << std::endl;
        return 1;
    }

    return 0;
}
