/*
 * =====================================================================================
 *
 * Description:
 *  - Maps M tasks to P processing nodes (P << M) to minimize Total Cost.
 *  - Total Cost = Execution Cost + Inter-task Communication Cost.
 *  - Inter-task communication cost between tasks i and j is incurred ONLY when
 *    assignment[i] != assignment[j] (tasks are on different nodes).
 *  - Compares Sequential PSO vs. Parallel OpenMP PSO.
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// Problem Configuration
#define M               300     // Number of tasks
#define P               10      // Number of processing nodes (P << M)
#define SWARM_SIZE      100     // Number of particles in the swarm
#define MAX_ITER        100     // Maximum number of iterations

// PSO Hyperparameters
#define W               0.72    // Inertia weight
#define C1              1.49    // Cognitive acceleration coefficient (personal best)
#define C2              1.49    // Social acceleration coefficient (global best)
#define V_MAX           4.0     // Maximum velocity clamp

// Global cost matrices
double E[M][P];                 // Execution cost matrix: E[task][node]
double C[M][M];                 // Communication cost matrix: C[task1][task2]

// Particle structure
typedef struct {
    double pos[M];              // Continuous position (in [0, P))
    double vel[M];              // Velocity
    int assignment[M];          // Discrete task assignment: (int)floor(pos[k])
    double pbest_pos[M];        // Personal best position
    double pbest_cost;          // Personal best cost
    double current_cost;        // Current cost
} Particle;

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
    srand(42);

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

// Map continuous position to discrete node index in [0, P-1]
int map_to_node(double val) {
    int node = (int)floor(val);
    if (node < 0) node = 0;
    if (node >= P) node = P - 1;
    return node;
}

// =====================================================================================
// 1. SEQUENTIAL PARTICLE SWARM OPTIMIZATION
// =====================================================================================
double run_sequential_pso(int *best_sol) {
    Particle swarm[SWARM_SIZE];
    double gbest_pos[M];
    double gbest_cost = 1e12;

    // Initialize particles
    for (int i = 0; i < SWARM_SIZE; i++) {
        for (int k = 0; k < M; k++) {
            swarm[i].pos[k] = ((double)rand() / RAND_MAX) * P;
            swarm[i].vel[k] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * V_MAX * 0.5;
            swarm[i].pbest_pos[k] = swarm[i].pos[k];
            swarm[i].assignment[k] = map_to_node(swarm[i].pos[k]);
        }
        swarm[i].current_cost = calculate_cost(swarm[i].assignment);
        swarm[i].pbest_cost = swarm[i].current_cost;

        if (swarm[i].pbest_cost < gbest_cost) {
            gbest_cost = swarm[i].pbest_cost;
            memcpy(gbest_pos, swarm[i].pbest_pos, sizeof(double) * M);
        }
    }

    double start_time = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < SWARM_SIZE; i++) {
            for (int k = 0; k < M; k++) {
                double r1 = (double)rand() / RAND_MAX;
                double r2 = (double)rand() / RAND_MAX;

                // Update velocity
                double v = W * swarm[i].vel[k] +
                           C1 * r1 * (swarm[i].pbest_pos[k] - swarm[i].pos[k]) +
                           C2 * r2 * (gbest_pos[k] - swarm[i].pos[k]);

                if (v > V_MAX) v = V_MAX;
                if (v < -V_MAX) v = -V_MAX;
                swarm[i].vel[k] = v;

                // Update position
                double p = swarm[i].pos[k] + v;
                if (p < 0.0) p = 0.0;
                if (p >= P) p = P - 1e-4;
                swarm[i].pos[k] = p;

                swarm[i].assignment[k] = map_to_node(p);
            }

            // Evaluate fitness
            swarm[i].current_cost = calculate_cost(swarm[i].assignment);

            // Update personal best
            if (swarm[i].current_cost < swarm[i].pbest_cost) {
                swarm[i].pbest_cost = swarm[i].current_cost;
                memcpy(swarm[i].pbest_pos, swarm[i].pos, sizeof(double) * M);

                // Update global best
                if (swarm[i].pbest_cost < gbest_cost) {
                    gbest_cost = swarm[i].pbest_cost;
                    memcpy(gbest_pos, swarm[i].pbest_pos, sizeof(double) * M);
                }
            }
        }
    }

    double end_time = omp_get_wtime();

    for (int k = 0; k < M; k++) {
        best_sol[k] = map_to_node(gbest_pos[k]);
    }

    return end_time - start_time;
}

// Simple thread-safe random number generator (ANSI C standard)
static inline double simple_rand_double(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (double)((*seed / 65536) % 32768) / 32768.0;
}

// =====================================================================================
// 2. PARALLEL PARTICLE SWARM OPTIMIZATION (OpenMP)
// =====================================================================================
double run_parallel_pso(int *best_sol) {
    Particle swarm[SWARM_SIZE];
    double gbest_pos[M];
    double gbest_cost = 1e12;

    // Parallel initialization of particles
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < SWARM_SIZE; i++) {
        unsigned int seed = 42 + i * 29;
        for (int k = 0; k < M; k++) {
            swarm[i].pos[k] = simple_rand_double(&seed) * P;
            swarm[i].vel[k] = (simple_rand_double(&seed) * 2.0 - 1.0) * V_MAX * 0.5;
            swarm[i].pbest_pos[k] = swarm[i].pos[k];
            swarm[i].assignment[k] = map_to_node(swarm[i].pos[k]);
        }
        swarm[i].current_cost = calculate_cost(swarm[i].assignment);
        swarm[i].pbest_cost = swarm[i].current_cost;
    }

    // Find initial global best
    for (int i = 0; i < SWARM_SIZE; i++) {
        if (swarm[i].pbest_cost < gbest_cost) {
            gbest_cost = swarm[i].pbest_cost;
            memcpy(gbest_pos, swarm[i].pbest_pos, sizeof(double) * M);
        }
    }

    double start_time = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Parallel swarm updates and fitness evaluations
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < SWARM_SIZE; i++) {
            unsigned int seed = iter * 1000 + i * 19 + omp_get_thread_num();

            for (int k = 0; k < M; k++) {
                double r1 = simple_rand_double(&seed);
                double r2 = simple_rand_double(&seed);

                // Update velocity
                double v = W * swarm[i].vel[k] +
                           C1 * r1 * (swarm[i].pbest_pos[k] - swarm[i].pos[k]) +
                           C2 * r2 * (gbest_pos[k] - swarm[i].pos[k]);

                if (v > V_MAX) v = V_MAX;
                if (v < -V_MAX) v = -V_MAX;
                swarm[i].vel[k] = v;

                // Update position
                double p = swarm[i].pos[k] + v;
                if (p < 0.0) p = 0.0;
                if (p >= P) p = P - 1e-4;
                swarm[i].pos[k] = p;

                swarm[i].assignment[k] = map_to_node(p);
            }

            // Parallel fitness evaluation
            swarm[i].current_cost = calculate_cost(swarm[i].assignment);

            // Update personal best
            if (swarm[i].current_cost < swarm[i].pbest_cost) {
                swarm[i].pbest_cost = swarm[i].current_cost;
                memcpy(swarm[i].pbest_pos, swarm[i].pos, sizeof(double) * M);
            }
        }

        // Global best reduction
        for (int i = 0; i < SWARM_SIZE; i++) {
            if (swarm[i].pbest_cost < gbest_cost) {
                gbest_cost = swarm[i].pbest_cost;
                memcpy(gbest_pos, swarm[i].pbest_pos, sizeof(double) * M);
            }
        }
    }

    double end_time = omp_get_wtime();

    for (int k = 0; k < M; k++) {
        best_sol[k] = map_to_node(gbest_pos[k]);
    }

    return end_time - start_time;
}

// =====================================================================================
// MAIN FUNCTION
// =====================================================================================
int main() {
    int best_seq[M];
    int best_par[M];

    printf("====================================================================\n");
    printf(" HPC LAB 5: Task Allocation using Particle Swarm Optimization (PSO)\n");
    printf(" Tasks (M): %d | Processing Nodes (P): %d | Threads: %d\n", M, P, omp_get_max_threads());
    printf(" Swarm Size: %d | Iterations: %d\n", SWARM_SIZE, MAX_ITER);
    printf("====================================================================\n\n");

    // Initialize problem matrices
    init_problem_data();

    // 1. Run Sequential PSO
    printf("Running Sequential Particle Swarm Optimization...\n");
    double time_seq = run_sequential_pso(best_seq);
    double cost_seq = calculate_cost(best_seq);
    printf("  Sequential PSO Completed in : %8.4f seconds | Best Cost: %.2f\n\n", time_seq, cost_seq);

    // 2. Run Parallel PSO (OpenMP)
    printf("Running Parallel Particle Swarm Optimization (OpenMP)...\n");
    double time_par = run_parallel_pso(best_par);
    double cost_par = calculate_cost(best_par);
    printf("  Parallel PSO Completed in   : %8.4f seconds | Best Cost: %.2f\n\n", time_par, cost_par);

    // 3. Performance Comparison Table
    double speedup = time_seq / time_par;
    double efficiency = (speedup / omp_get_max_threads()) * 100.0;

    printf("+-----------------------------------+---------------+------------+------------+\n");
    printf("| Implementation                    | Time (sec)    | Best Cost  | Speedup    |\n");
    printf("+-----------------------------------+---------------+------------+------------+\n");
    printf("| Sequential Particle Swarm (SPSO)  | %9.4f s   | %10.2f | %9.2fx |\n", time_seq, cost_seq, 1.00);
    printf("| Parallel Particle Swarm (OMP)     | %9.4f s   | %10.2f | %9.2fx |\n", time_par, cost_par, speedup);
    printf("+-----------------------------------+---------------+------------+------------+\n\n");

    printf("====================================================================\n");
    printf(" Speedup: %.2fx | Parallel Efficiency: %.2f%%\n", speedup, efficiency);
    printf("====================================================================\n");

    return 0;
}
