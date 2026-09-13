/*
 * OpenMP Sections - Basic Example
 * 
 * The 'sections' construct (#pragma omp sections) is a work-sharing
 * construct used to distribute independent, non-iterative blocks of code
 * across threads in a team. Each #pragma omp section is executed exactly
 * once by one thread.
 * 
 * Key Characteristics:
 *  1. #pragma omp sections: Enclosing block defining a set of sections.
 *  2. #pragma omp section: Defines an individual block of work.
 *  3. Dynamic assignment: Threads pick up available sections; if there are
 *     more sections than threads, threads execute remaining sections as
 *     they finish their current one.
 *  4. Implicit Barrier: All threads synchronize at the end of the 
 *     sections block (unless the 'nowait' clause is specified).
 * 
 * Compilation & Execution:
 *   gcc -fopenmp sections.c -o sections.exe
 *   .\sections.exe
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

#define ARRAY_SIZE 10000000

// -------------------------------------------------------------
// Example 1: Independent Operations on a Shared Dataset
// -------------------------------------------------------------
// Different threads execute completely different analytical tasks
// on the same array concurrently.
void example_sections(int *data, int n) {
    long long sum = 0;
    int min_val = data[0];
    int max_val = data[0];
    double avg = 0.0;

    printf("=== Example 1: Parallel Work-Sharing with Sections ===\n");

    #pragma omp parallel num_threads(3)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                printf("[Section 1: Sum & Avg] Executed by Thread %d\n", omp_get_thread_num());
                for (int i = 0; i < n; i++) {
                    sum += data[i];
                }
                avg = (double)sum / n;
            }

            #pragma omp section
            {
                printf("[Section 2: Find Minimum] Executed by Thread %d\n", omp_get_thread_num());
                for (int i = 1; i < n; i++) {
                    if (data[i] < min_val) {
                        min_val = data[i];
                    }
                }
            }

            #pragma omp section
            {
                printf("[Section 3: Find Maximum] Executed by Thread %d\n", omp_get_thread_num());
                for (int i = 1; i < n; i++) {
                    if (data[i] > max_val) {
                        max_val = data[i];
                    }
                }
            }
        } // Implicit barrier here: all threads wait until all sections finish
    }

    printf("\nResults:\n");
    printf("  Sum:     %lld\n", sum);
    printf("  Average: %.2f\n", avg);
    printf("  Minimum: %d\n", min_val);
    printf("  Maximum: %d\n", max_val);
}

// -------------------------------------------------------------
// Example 2: Combined Directive (#pragma omp parallel sections)
// -------------------------------------------------------------
// When sections is the only construct inside parallel, you can
// use the combined '#pragma omp parallel sections' shorthand.
void example_parallel_sections_shorthand(void) {
    printf("\n=== Example 2: Combined '#pragma omp parallel sections' ===\n");

    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section
        {
            printf("Download subtask: Thread %d is fetching data...\n", omp_get_thread_num());
            sleep_ms(30);
            printf("Download subtask: Thread %d finished fetching.\n", omp_get_thread_num());
        }

        #pragma omp section
        {
            printf("Preprocessing subtask: Thread %d is parsing configs...\n", omp_get_thread_num());
            sleep_ms(20);
            printf("Preprocessing subtask: Thread %d finished parsing.\n", omp_get_thread_num());
        }

        #pragma omp section
        {
            printf("Logging subtask: Thread %d is initializing logs...\n", omp_get_thread_num());
            sleep_ms(10);
            printf("Logging subtask: Thread %d finished logging.\n", omp_get_thread_num());
        }
    }
}

int main(void) {
    int *data = (int *)malloc(sizeof(int) * ARRAY_SIZE);
    if (!data) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Initialize array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        data[i] = (i % 1000) + 1; // Values from 1 to 1000
    }

    example_sections(data, ARRAY_SIZE);
    example_parallel_sections_shorthand();

    free(data);
    return 0;
}
