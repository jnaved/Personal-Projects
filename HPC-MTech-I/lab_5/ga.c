/*
 * =====================================================================================
 * Description:
 *  - Maps M tasks to P processing nodes (P << M) to minimize Total Cost.
 *  - Total Cost = Execution Cost + Inter-task Communication Cost.
 *  - Inter-task communication cost between tasks i and j is incurred ONLY when
 *    assignment[i] != assignment[j] (tasks are on different nodes).
 *  - Compares Sequential Genetic Algorithm vs. Parallel OpenMP Genetic Algorithm.
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Problem Configuration
#define M               300     // Number of tasks
#define P               10      // Number of processing nodes (P << M)
#define POP_SIZE        100     // Number of individuals in population
#define MAX_GEN         100     // Number of generations
#define CROSSOVER_RATE  0.85    // Probability of crossover
#define MUTATION_RATE   0.02    // Probability of mutation per gene

// Global cost matrices
double E[M][P];                 // Execution cost matrix: E[task][node]
double C[M][M];                 // Communication cost matrix: C[task1][task2]

// Structure representing a candidate assignment
typedef struct {
    int assignment[M];          // assignment[i] gives the node (0 to P-1) for task i
    double cost;                // Total cost (lower is better)
} Individual;

// =====================================================================================
// COST CALCULATION
// =====================================================================================
double calculate_cost(const int *assignment) {
    double total_exec = 0.0;
    double total_comm = 0.0;

    // 1. Total Execution Cost = sum of E[i][assignment[i]]
    for (int i = 0; i < M; i++) {
        total_exec += E[i][assignment[i]];
    }

    // 2. Inter-task Communication Cost: incurred only when tasks are on different nodes
    for (int i = 0; i < M; i++) {
        for (int j = i + 1; j < M; j++) {
            if (assignment[i] != assignment[j]) {
                total_comm += C[i][j];
            }
        }
    }

    return total_exec + total_comm;
}

// Initialize synthetic execution and communication cost matrices
void init_problem_data() {
    srand(42); // Fixed seed for reproducible benchmarks

    // Initialize execution costs (range: 10 to 80)
    for (int i = 0; i < M; i++) {
        for (int p = 0; p < P; p++) {
            E[i][p] = 10.0 + (rand() % 71);
        }
    }

    // Initialize communication costs (symmetric, range: 0 to 40, ~25% density)
    for (int i = 0; i < M; i++) {
        C[i][i] = 0.0;
        for (int j = i + 1; j < M; j++) {
            if ((rand() % 100) < 25) {
                double cost = 5.0 + (rand() % 36);
                C[i][j] = cost;
                C[j][i] = cost;
            } else {
                C[i][j] = 0.0;
                C[j][i] = 0.0;
            }
        }
    }
}

// Compare function for sorting population by cost (ascending)
int compare_individuals(const void *a, const void *b) {
    Individual *indA = (Individual *)a;
    Individual *indB = (Individual *)b;
    if (indA->cost < indB->cost) return -1;
    if (indA->cost > indB->cost) return 1;
    return 0;
}

// =====================================================================================
// 1. SEQUENTIAL GENETIC ALGORITHM
// =====================================================================================
double run_sequential_ga(int *best_sol) {
    Individual pop[POP_SIZE];
    Individual next_pop[POP_SIZE];

    // Initialize population randomly
    for (int i = 0; i < POP_SIZE; i++) {
        for (int k = 0; k < M; k++) {
            pop[i].assignment[k] = rand() % P;
        }
        pop[i].cost = calculate_cost(pop[i].assignment);
    }

    double start_time = omp_get_wtime();

    for (int gen = 0; gen < MAX_GEN; gen++) {
        // Sort to find best individuals (elitism)
        qsort(pop, POP_SIZE, sizeof(Individual), compare_individuals);

        // Elitism: carry forward the 2 best individuals unchanged
        next_pop[0] = pop[0];
        next_pop[1] = pop[1];

        // Generate offspring
        for (int i = 2; i < POP_SIZE; i++) {
            // Tournament selection (choose best among 3 random individuals)
            int p1 = rand() % POP_SIZE;
            for (int t = 1; t < 3; t++) {
                int cand = rand() % POP_SIZE;
                if (pop[cand].cost < pop[p1].cost) p1 = cand;
            }

            int p2 = rand() % POP_SIZE;
            for (int t = 1; t < 3; t++) {
                int cand = rand() % POP_SIZE;
                if (pop[cand].cost < pop[p2].cost) p2 = cand;
            }

            // Single-point Crossover
            if ((double)rand() / RAND_MAX < CROSSOVER_RATE) {
                int cross_point = rand() % M;
                for (int k = 0; k < cross_point; k++) next_pop[i].assignment[k] = pop[p1].assignment[k];
                for (int k = cross_point; k < M; k++) next_pop[i].assignment[k] = pop[p2].assignment[k];
            } else {
                for (int k = 0; k < M; k++) next_pop[i].assignment[k] = pop[p1].assignment[k];
            }

            // Mutation: randomly change task node
            for (int k = 0; k < M; k++) {
                if ((double)rand() / RAND_MAX < MUTATION_RATE) {
                    next_pop[i].assignment[k] = rand() % P;
                }
            }

            // Evaluate fitness
            next_pop[i].cost = calculate_cost(next_pop[i].assignment);
        }

        // Copy next generation to current
        memcpy(pop, next_pop, sizeof(Individual) * POP_SIZE);
    }

    double end_time = omp_get_wtime();

    qsort(pop, POP_SIZE, sizeof(Individual), compare_individuals);
    memcpy(best_sol, pop[0].assignment, sizeof(int) * M);

    return end_time - start_time;
}

// Simple thread-safe random number generator (ANSI C standard)
static inline int simple_rand(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (int)((*seed / 65536) % 32768);
}

// =====================================================================================
// 2. PARALLEL GENETIC ALGORITHM (OpenMP)
// =====================================================================================
double run_parallel_ga(int *best_sol) {
    Individual pop[POP_SIZE];
    Individual next_pop[POP_SIZE];

    // Parallel population initialization
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < POP_SIZE; i++) {
        unsigned int seed = 42 + i * 17;
        for (int k = 0; k < M; k++) {
            pop[i].assignment[k] = simple_rand(&seed) % P;
        }
        pop[i].cost = calculate_cost(pop[i].assignment);
    }

    double start_time = omp_get_wtime();

    for (int gen = 0; gen < MAX_GEN; gen++) {
        // Elitism: sort to find best solutions
        qsort(pop, POP_SIZE, sizeof(Individual), compare_individuals);

        next_pop[0] = pop[0];
        next_pop[1] = pop[1];

        // Parallel reproduction and fitness evaluation across population
        #pragma omp parallel for schedule(static)
        for (int i = 2; i < POP_SIZE; i++) {
            unsigned int seed = gen * 1000 + i * 31 + omp_get_thread_num();

            // Tournament selection
            int p1 = simple_rand(&seed) % POP_SIZE;
            for (int t = 1; t < 3; t++) {
                int cand = simple_rand(&seed) % POP_SIZE;
                if (pop[cand].cost < pop[p1].cost) p1 = cand;
            }

            int p2 = simple_rand(&seed) % POP_SIZE;
            for (int t = 1; t < 3; t++) {
                int cand = simple_rand(&seed) % POP_SIZE;
                if (pop[cand].cost < pop[p2].cost) p2 = cand;
            }

            // Single-point Crossover
            if ((double)simple_rand(&seed) / 32768.0 < CROSSOVER_RATE) {
                int cross_point = simple_rand(&seed) % M;
                for (int k = 0; k < cross_point; k++) next_pop[i].assignment[k] = pop[p1].assignment[k];
                for (int k = cross_point; k < M; k++) next_pop[i].assignment[k] = pop[p2].assignment[k];
            } else {
                for (int k = 0; k < M; k++) next_pop[i].assignment[k] = pop[p1].assignment[k];
            }

            // Mutation
            for (int k = 0; k < M; k++) {
                if ((double)simple_rand(&seed) / 32768.0 < MUTATION_RATE) {
                    next_pop[i].assignment[k] = simple_rand(&seed) % P;
                }
            }

            // Parallel fitness evaluation (the most computation-intensive step)
            next_pop[i].cost = calculate_cost(next_pop[i].assignment);
        }

        // Copy next generation to current
        memcpy(pop, next_pop, sizeof(Individual) * POP_SIZE);
    }

    double end_time = omp_get_wtime();

    qsort(pop, POP_SIZE, sizeof(Individual), compare_individuals);
    memcpy(best_sol, pop[0].assignment, sizeof(int) * M);

    return end_time - start_time;
}

// =====================================================================================
// MAIN FUNCTION
// =====================================================================================
int main() {
    int best_seq[M];
    int best_par[M];

    printf("====================================================================\n");
    printf(" HPC LAB 5: Task Allocation using Genetic Algorithm (GA)\n");
    printf(" Tasks (M): %d | Processing Nodes (P): %d | Threads: %d\n", M, P, omp_get_max_threads());
    printf(" Population Size: %d | Generations: %d\n", POP_SIZE, MAX_GEN);
    printf("====================================================================\n\n");

    // Initialize problem matrices
    init_problem_data();

    // 1. Run Sequential GA
    printf("Running Sequential Genetic Algorithm...\n");
    double time_seq = run_sequential_ga(best_seq);
    double cost_seq = calculate_cost(best_seq);
    printf("  Sequential GA Completed in : %8.4f seconds | Best Cost: %.2f\n\n", time_seq, cost_seq);

    // 2. Run Parallel GA (OpenMP)
    printf("Running Parallel Genetic Algorithm (OpenMP)...\n");
    double time_par = run_parallel_ga(best_par);
    double cost_par = calculate_cost(best_par);
    printf("  Parallel GA Completed in   : %8.4f seconds | Best Cost: %.2f\n\n", time_par, cost_par);

    // 3. Performance Comparison Table
    double speedup = time_seq / time_par;
    double efficiency = (speedup / omp_get_max_threads()) * 100.0;

    printf("+-----------------------------------+---------------+------------+------------+\n");
    printf("| Implementation                    | Time (sec)    | Best Cost  | Speedup    |\n");
    printf("+-----------------------------------+---------------+------------+------------+\n");
    printf("| Sequential Genetic Algorithm      | %9.4f s   | %10.2f | %9.2fx |\n", time_seq, cost_seq, 1.00);
    printf("| Parallel Genetic Algorithm (OMP)  | %9.4f s   | %10.2f | %9.2fx |\n", time_par, cost_par, speedup);
    printf("+-----------------------------------+---------------+------------+------------+\n\n");

    printf("====================================================================\n");
    printf(" Speedup: %.2fx | Parallel Efficiency: %.2f%%\n", speedup, efficiency);
    printf("====================================================================\n");

    return 0;
}
