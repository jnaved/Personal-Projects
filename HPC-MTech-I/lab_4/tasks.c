/*
 * OpenMP Tasking - Basic Example
 * 
 * OpenMP tasks (#pragma omp task) allow parallelizing irregular,
 * dynamic, or recursive workloads that standard work-sharing loops
 * (#pragma omp for) cannot easily handle.
 * 
 * Standard Tasking Pattern:
 *  1. #pragma omp parallel: Creates a team of threads.
 *  2. #pragma omp single: Ensures only ONE thread produces/dispatches tasks.
 *  3. #pragma omp task: Wraps an independent unit of work. Any available
 *     thread in the team can pick up and execute this task.
 *  4. #pragma omp taskwait: Waits for all direct child tasks to finish.
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <unistd.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

// -------------------------------------------------------------
// Example 1: Basic Loop of Dynamic Tasks
// -------------------------------------------------------------
void example_basic_tasks(void) {
    printf("=== Example 1: Basic Independent Tasks ===\n");

    #pragma omp parallel num_threads(4)
    {
        // A single thread creates the tasks, all 4 threads execute them
        #pragma omp single
        {
            printf("Thread %d is generating tasks...\n\n", omp_get_thread_num());

            for (int i = 1; i <= 8; i++) {
                // 'firstprivate(i)' ensures each task gets its own copy of 'i'
                #pragma omp task firstprivate(i)
                {
                    printf("[Task %d] Started by thread %d\n", i, omp_get_thread_num());
                    
                    // Simulate some work
                    sleep_ms(50);

                    printf("[Task %d] Finished by thread %d\n", i, omp_get_thread_num());
                }
            }

            // Optional explicit barrier for tasks created in this single block.
            // (Note: #pragma omp single has an implicit barrier at the end,
            //  but #pragma omp taskwait explicitly waits for child tasks here).
            #pragma omp taskwait
            printf("\nAll 8 tasks have completed!\n");
        }
    }
}

// -------------------------------------------------------------
// Example 2: Recursive Tasking with taskwait (Fibonacci)
// -------------------------------------------------------------
// Recursive problems are the primary use case for OpenMP tasks.
long long fib(int n) {
    if (n < 2) return n;

    // For small n, avoid task-creation overhead by running sequentially
    if (n < 20) {
        return fib(n - 1) + fib(n - 2);
    }

    long long x, y;

    // Spawn task for fib(n - 1)
    #pragma omp task shared(x)
    x = fib(n - 1);

    // Spawn task for fib(n - 2)
    #pragma omp task shared(y)
    y = fib(n - 2);

    // Wait until both child tasks finish before computing (x + y)
    #pragma omp taskwait

    return x + y;
}

void example_recursive_tasks(void) {
    printf("\n=== Example 2: Recursive Tasks (Fibonacci) ===\n");
    int n = 30;
    long long result = 0;

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            result = fib(n);
        }
    }

    double end = omp_get_wtime();

    printf("Fibonacci(%d) = %lld\n", n, result);
    printf("Computed in %.4f seconds using OpenMP tasks.\n", end - start);
}

int main(void) {
    example_basic_tasks();
    example_recursive_tasks();
    return 0;
}
