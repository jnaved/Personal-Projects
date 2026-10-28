/*
 * =====================================================================================
 * HPC Lab 7: Task Allocation Problem (TAP) using Metaheuristics
 * Comparison of Sequential, OpenMP, and CUDA C Implementations
 * 
 * Metaheuristics:
 *  1. Genetic Algorithm (GA)
 *  2. Particle Swarm Optimization (PSO)
 *
 * Problem Specification (from Lab Assignment 5):
 *  - M = 300 tasks to be assigned to P = 10 processing nodes (P << M).
 *  - Total Cost = Execution Cost + Inter-task Communication Cost.
 *  - Communication cost between task i and j is incurred ONLY when 
 *    assignment[i] != assignment[j] (tasks are on different nodes).
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

// Problem Configuration
#define M               300     // Number of tasks
#define P               10      // Number of processing nodes
#define POP_SIZE        100     // Population size for GA
#define MAX_GEN         100     // Generations for GA
#define CROSSOVER_RATE  0.85    // Crossover probability
#define MUTATION_RATE   0.02    // Mutation probability per gene

#define SWARM_SIZE      100     // Number of particles for PSO
#define MAX_ITER        100     // Iterations for PSO
#define W               0.72    // PSO Inertia weight
#define C1              1.49    // PSO Cognitive coefficient
#define C2              1.49    // PSO Social coefficient
#define V_MAX           4.0     // PSO Velocity limit

// Host problem matrices
double h_E[M][P];               // Execution cost matrix: E[task][node]
double h_C[M][M];               // Communication cost matrix: C[task1][task2]

// Device problem matrices
double *d_E = NULL;
double *d_C = NULL;

// Macro for CUDA error checking
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Initialize problem data identically to Lab 5
void init_problem_data() {
    cudaFree(0); // Warmup CUDA runtime driver context
    srand(42);   // Fixed seed for reproducibility

    // Execution costs (10.0 to 80.0)
    for (int i = 0; i < M; i++) {
        for (int p = 0; p < P; p++) {
            h_E[i][p] = 10.0 + (rand() % 71);
        }
    }

    // Communication costs (symmetric, range 5.0 to 40.0, ~25% density)
    for (int i = 0; i < M; i++) {
        h_C[i][i] = 0.0;
        for (int j = i + 1; j < M; j++) {
            if ((rand() % 100) < 25) {
                double cost = 5.0 + (rand() % 36);
                h_C[i][j] = cost;
                h_C[j][i] = cost;
            } else {
                h_C[i][j] = 0.0;
                h_C[j][i] = 0.0;
            }
        }
    }

    // Allocate and copy matrices to CUDA device
    CHECK_CUDA(cudaMalloc(&d_E, M * P * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, M * M * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_E, h_E, M * P * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M * M * sizeof(double), cudaMemcpyHostToDevice));
}

void cleanup_problem_data() {
    if (d_E) cudaFree(d_E);
    if (d_C) cudaFree(d_C);
}

// Host cost calculation
double calculate_cost_host(const int *assignment) {
    double total_exec = 0.0;
    double total_comm = 0.0;

    for (int i = 0; i < M; i++) {
        total_exec += h_E[i][assignment[i]];
    }

    for (int i = 0; i < M; i++) {
        int a_i = assignment[i];
        for (int j = i + 1; j < M; j++) {
            if (a_i != assignment[j]) {
                total_comm += h_C[i][j];
            }
        }
    }

    return total_exec + total_comm;
}

// Device parallel fitness evaluation kernel (1 block of 128 threads per candidate solution)
__global__ void evaluate_tap_block_kernel(const int *solutions, double *costs,
                                          const double *E, const double *C, int num_sols) {
    int ind = blockIdx.x;
    if (ind >= num_sols) return;

    const int *assignment = &solutions[ind * M];
    double local_sum = 0.0;

    // Parallel execution cost across threads in block
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        local_sum += E[i * P + assignment[i]];
    }

    // Parallel communication cost across threads in block
    for (int i = 0; i < M; i++) {
        int a_i = assignment[i];
        for (int j = i + 1 + threadIdx.x; j < M; j += blockDim.x) {
            if (a_i != assignment[j]) {
                local_sum += C[i * M + j];
            }
        }
    }

    // Block-level warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    __shared__ double warp_sums[4]; // 128 threads / 32 = 4 warps
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        double val = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[threadIdx.x] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
            costs[ind] = val;
        }
    }
}

// Helper PRNGs
static inline int simple_rand(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (int)((*seed / 65536) % 32768);
}

static inline double simple_rand_double(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (double)((*seed / 65536) % 32768) / 32768.0;
}

__device__ inline unsigned int gpu_rand_int(unsigned int *seed) {
    *seed = *seed * 1103515245ULL + 12345ULL;
    return (*seed / 65536) % 32768;
}

__device__ inline double gpu_rand_double(unsigned int *seed) {
    *seed = *seed * 1103515245ULL + 12345ULL;
    return (double)((*seed / 65536) % 32768) / 32768.0;
}

// Individual structure for host GA
typedef struct {
    int assignment[M];
    double cost;
} Individual;

int compare_individuals(const void *a, const void *b) {
    Individual *indA = (Individual *)a;
    Individual *indB = (Individual *)b;
    if (indA->cost < indB->cost) return -1;
    if (indA->cost > indB->cost) return 1;
    return 0;
}

// =====================================================================================
// 1. GENETIC ALGORITHM - SEQUENTIAL (from Lab 5)
// =====================================================================================
double run_sequential_ga(int *best_sol, double *best_cost) {
    Individual pop[POP_SIZE];
    Individual next_pop[POP_SIZE];

    for (int i = 0; i < POP_SIZE; i++) {
        for (int k = 0; k < M; k++) {
            pop[i].assignment[k] = rand() % P;
        }
        pop[i].cost = calculate_cost_host(pop[i].assignment);
    }

    double start_time = omp_get_wtime();

    for (int gen = 0; gen < MAX_GEN; gen++) {
        qsort(pop, POP_SIZE, sizeof(Individual), compare_individuals);

        next_pop[0] = pop[0];
        next_pop[1] = pop[1];

        for (int i = 2; i < POP_SIZE; i++) {
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

            if ((double)rand() / RAND_MAX < CROSSOVER_RATE) {
                int cross_point = rand() % M;
                for (int k = 0; k < cross_point; k++) next_pop[i].assignment[k] = pop[p1].assignment[k];
                for (int k = cross_point; k < M; k++) next_pop[i].assignment[k] = pop[p2].assignment[k];
            } else {
                for (int k = 0; k < M; k++) next_pop[i].assignment[k] = pop[p1].assignment[k];
            }

            for (int k = 0; k < M; k++) {
                if ((double)rand() / RAND_MAX < MUTATION_RATE) {
                    next_pop[i].assignment[k] = rand() % P;
                }
            }

            next_pop[i].cost = calculate_cost_host(next_pop[i].assignment);
        }

        memcpy(pop, next_pop, sizeof(Individual) * POP_SIZE);
    }

    double end_time = omp_get_wtime();

    qsort(pop, POP_SIZE, sizeof(Individual), compare_individuals);
    memcpy(best_sol, pop[0].assignment, sizeof(int) * M);
    *best_cost = pop[0].cost;

    return end_time - start_time;
}

// =====================================================================================
// 2. GENETIC ALGORITHM - OPENMP (from Lab 5)
// =====================================================================================
double run_openmp_ga(int *best_sol, double *best_cost) {
    Individual pop[POP_SIZE];
    Individual next_pop[POP_SIZE];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < POP_SIZE; i++) {
        unsigned int seed = 42 + i * 17;
        for (int k = 0; k < M; k++) {
            pop[i].assignment[k] = simple_rand(&seed) % P;
        }
        pop[i].cost = calculate_cost_host(pop[i].assignment);
    }

    double start_time = omp_get_wtime();

    for (int gen = 0; gen < MAX_GEN; gen++) {
        qsort(pop, POP_SIZE, sizeof(Individual), compare_individuals);

        next_pop[0] = pop[0];
        next_pop[1] = pop[1];

        #pragma omp parallel for schedule(static)
        for (int i = 2; i < POP_SIZE; i++) {
            unsigned int seed = gen * 1000 + i * 31 + omp_get_thread_num();

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

            if ((double)simple_rand(&seed) / 32768.0 < CROSSOVER_RATE) {
                int cross_point = simple_rand(&seed) % M;
                for (int k = 0; k < cross_point; k++) next_pop[i].assignment[k] = pop[p1].assignment[k];
                for (int k = cross_point; k < M; k++) next_pop[i].assignment[k] = pop[p2].assignment[k];
            } else {
                for (int k = 0; k < M; k++) next_pop[i].assignment[k] = pop[p1].assignment[k];
            }

            for (int k = 0; k < M; k++) {
                if ((double)simple_rand(&seed) / 32768.0 < MUTATION_RATE) {
                    next_pop[i].assignment[k] = simple_rand(&seed) % P;
                }
            }

            next_pop[i].cost = calculate_cost_host(next_pop[i].assignment);
        }

        memcpy(pop, next_pop, sizeof(Individual) * POP_SIZE);
    }

    double end_time = omp_get_wtime();

    qsort(pop, POP_SIZE, sizeof(Individual), compare_individuals);
    memcpy(best_sol, pop[0].assignment, sizeof(int) * M);
    *best_cost = pop[0].cost;

    return end_time - start_time;
}

// =====================================================================================
// 3. GENETIC ALGORITHM - CUDA C
// =====================================================================================
__global__ void ga_init_pop_kernel(int *pop, unsigned int *seeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POP_SIZE) return;

    unsigned int seed = seeds[i];
    for (int k = 0; k < M; k++) {
        pop[i * M + k] = gpu_rand_int(&seed) % P;
    }
    seeds[i] = seed;
}

__global__ void ga_find_elites_kernel(const double *costs, int *elite_indices) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int b1 = 0, b2 = 1;
        if (costs[b2] < costs[b1]) {
            b1 = 1; b2 = 0;
        }
        for (int i = 2; i < POP_SIZE; i++) {
            if (costs[i] < costs[b1]) {
                b2 = b1;
                b1 = i;
            } else if (costs[i] < costs[b2]) {
                b2 = i;
            }
        }
        elite_indices[0] = b1;
        elite_indices[1] = b2;
    }
}

__global__ void ga_reproduce_kernel(const int *pop, const double *costs,
                                    int *next_pop,
                                    const int *elite_indices, unsigned int *seeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POP_SIZE) return;

    // First two slots are preserved for elites
    if (i == 0 || i == 1) {
        int elite = elite_indices[i];
        for (int k = 0; k < M; k++) {
            next_pop[i * M + k] = pop[elite * M + k];
        }
        return;
    }

    unsigned int seed = seeds[i];

    // Tournament selection 1
    int p1 = gpu_rand_int(&seed) % POP_SIZE;
    for (int t = 1; t < 3; t++) {
        int cand = gpu_rand_int(&seed) % POP_SIZE;
        if (costs[cand] < costs[p1]) p1 = cand;
    }

    // Tournament selection 2
    int p2 = gpu_rand_int(&seed) % POP_SIZE;
    for (int t = 1; t < 3; t++) {
        int cand = gpu_rand_int(&seed) % POP_SIZE;
        if (costs[cand] < costs[p2]) p2 = cand;
    }

    // Single-point Crossover
    if (gpu_rand_double(&seed) < CROSSOVER_RATE) {
        int cross_point = gpu_rand_int(&seed) % M;
        for (int k = 0; k < cross_point; k++) next_pop[i * M + k] = pop[p1 * M + k];
        for (int k = cross_point; k < M; k++) next_pop[i * M + k] = pop[p2 * M + k];
    } else {
        for (int k = 0; k < M; k++) next_pop[i * M + k] = pop[p1 * M + k];
    }

    // Mutation
    for (int k = 0; k < M; k++) {
        if (gpu_rand_double(&seed) < MUTATION_RATE) {
            next_pop[i * M + k] = gpu_rand_int(&seed) % P;
        }
    }

    seeds[i] = seed;
}

double run_cuda_ga(int *best_sol, double *best_cost) {
    int *d_pop, *d_next_pop;
    double *d_costs;
    unsigned int *d_seeds;
    int *d_elite_indices;

    CHECK_CUDA(cudaMalloc(&d_pop, POP_SIZE * M * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_next_pop, POP_SIZE * M * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_costs, POP_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_seeds, POP_SIZE * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_elite_indices, 2 * sizeof(int)));

    unsigned int h_seeds[POP_SIZE];
    for (int i = 0; i < POP_SIZE; i++) h_seeds[i] = 42 + i * 17;
    CHECK_CUDA(cudaMemcpy(d_seeds, h_seeds, POP_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice));

    int blockSize = 64;
    int numBlocks = (POP_SIZE + blockSize - 1) / blockSize;

    double start_time = omp_get_wtime();

    // 1. Initialize population on GPU
    ga_init_pop_kernel<<<numBlocks, blockSize>>>(d_pop, d_seeds);
    // 2. Initial evaluation (1 block of 128 threads per individual)
    evaluate_tap_block_kernel<<<POP_SIZE, 128>>>(d_pop, d_costs, d_E, d_C, POP_SIZE);

    for (int gen = 0; gen < MAX_GEN; gen++) {
        // Step 1: Find 2 best elite individuals
        ga_find_elites_kernel<<<1, 1>>>(d_costs, d_elite_indices);

        // Step 2: Reproduce into d_next_pop
        ga_reproduce_kernel<<<numBlocks, blockSize>>>(d_pop, d_costs, d_next_pop,
                                                      d_elite_indices, d_seeds);

        // Step 3: Fast parallel fitness evaluation of offspring
        evaluate_tap_block_kernel<<<POP_SIZE, 128>>>(d_next_pop, d_costs, d_E, d_C, POP_SIZE);

        // Step 4: Swap population pointers
        int *temp_pop = d_pop; d_pop = d_next_pop; d_next_pop = temp_pop;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double end_time = omp_get_wtime();

    // Find best solution from final population
    ga_find_elites_kernel<<<1, 1>>>(d_costs, d_elite_indices);
    int h_elites[2];
    CHECK_CUDA(cudaMemcpy(h_elites, d_elite_indices, 2 * sizeof(int), cudaMemcpyDeviceToHost));

    int best_idx = h_elites[0];
    CHECK_CUDA(cudaMemcpy(best_sol, &d_pop[best_idx * M], M * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(best_cost, &d_costs[best_idx], sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_pop);
    cudaFree(d_next_pop);
    cudaFree(d_costs);
    cudaFree(d_seeds);
    cudaFree(d_elite_indices);

    return end_time - start_time;
}

// Particle structure for host PSO
typedef struct {
    double pos[M];
    double vel[M];
    int assignment[M];
    double pbest_pos[M];
    double pbest_cost;
    double current_cost;
} Particle;

static inline int map_to_node(double val) {
    int node = (int)floor(val);
    if (node < 0) node = 0;
    if (node >= P) node = P - 1;
    return node;
}

__device__ inline int gpu_map_to_node(double val) {
    int node = (int)floor(val);
    if (node < 0) node = 0;
    if (node >= P) node = P - 1;
    return node;
}

// =====================================================================================
// 4. PARTICLE SWARM OPTIMIZATION - SEQUENTIAL (from Lab 5)
// =====================================================================================
double run_sequential_pso(int *best_sol, double *best_cost) {
    Particle swarm[SWARM_SIZE];
    double gbest_pos[M];
    double gbest_cost = 1e12;

    for (int i = 0; i < SWARM_SIZE; i++) {
        for (int k = 0; k < M; k++) {
            swarm[i].pos[k] = ((double)rand() / RAND_MAX) * P;
            swarm[i].vel[k] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * V_MAX * 0.5;
            swarm[i].pbest_pos[k] = swarm[i].pos[k];
            swarm[i].assignment[k] = map_to_node(swarm[i].pos[k]);
        }
        swarm[i].current_cost = calculate_cost_host(swarm[i].assignment);
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

                double v = W * swarm[i].vel[k] +
                           C1 * r1 * (swarm[i].pbest_pos[k] - swarm[i].pos[k]) +
                           C2 * r2 * (gbest_pos[k] - swarm[i].pos[k]);

                if (v > V_MAX) v = V_MAX;
                if (v < -V_MAX) v = -V_MAX;
                swarm[i].vel[k] = v;

                double p = swarm[i].pos[k] + v;
                if (p < 0.0) p = 0.0;
                if (p >= P) p = P - 1e-4;
                swarm[i].pos[k] = p;

                swarm[i].assignment[k] = map_to_node(p);
            }

            swarm[i].current_cost = calculate_cost_host(swarm[i].assignment);

            if (swarm[i].current_cost < swarm[i].pbest_cost) {
                swarm[i].pbest_cost = swarm[i].current_cost;
                memcpy(swarm[i].pbest_pos, swarm[i].pos, sizeof(double) * M);

                if (swarm[i].pbest_cost < gbest_cost) {
                    gbest_cost = swarm[i].pbest_cost;
                    memcpy(gbest_pos, swarm[i].pbest_pos, sizeof(double) * M);
                }
            }
        }
    }

    double end_time = omp_get_wtime();

    for (int k = 0; k < M; k++) best_sol[k] = map_to_node(gbest_pos[k]);
    *best_cost = gbest_cost;

    return end_time - start_time;
}

// =====================================================================================
// 5. PARTICLE SWARM OPTIMIZATION - OPENMP (from Lab 5)
// =====================================================================================
double run_openmp_pso(int *best_sol, double *best_cost) {
    Particle swarm[SWARM_SIZE];
    double gbest_pos[M];
    double gbest_cost = 1e12;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < SWARM_SIZE; i++) {
        unsigned int seed = 42 + i * 29;
        for (int k = 0; k < M; k++) {
            swarm[i].pos[k] = simple_rand_double(&seed) * P;
            swarm[i].vel[k] = (simple_rand_double(&seed) * 2.0 - 1.0) * V_MAX * 0.5;
            swarm[i].pbest_pos[k] = swarm[i].pos[k];
            swarm[i].assignment[k] = map_to_node(swarm[i].pos[k]);
        }
        swarm[i].current_cost = calculate_cost_host(swarm[i].assignment);
        swarm[i].pbest_cost = swarm[i].current_cost;
    }

    for (int i = 0; i < SWARM_SIZE; i++) {
        if (swarm[i].pbest_cost < gbest_cost) {
            gbest_cost = swarm[i].pbest_cost;
            memcpy(gbest_pos, swarm[i].pbest_pos, sizeof(double) * M);
        }
    }

    double start_time = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < SWARM_SIZE; i++) {
            unsigned int seed = iter * 1000 + i * 19 + omp_get_thread_num();

            for (int k = 0; k < M; k++) {
                double r1 = simple_rand_double(&seed);
                double r2 = simple_rand_double(&seed);

                double v = W * swarm[i].vel[k] +
                           C1 * r1 * (swarm[i].pbest_pos[k] - swarm[i].pos[k]) +
                           C2 * r2 * (gbest_pos[k] - swarm[i].pos[k]);

                if (v > V_MAX) v = V_MAX;
                if (v < -V_MAX) v = -V_MAX;
                swarm[i].vel[k] = v;

                double p = swarm[i].pos[k] + v;
                if (p < 0.0) p = 0.0;
                if (p >= P) p = P - 1e-4;
                swarm[i].pos[k] = p;

                swarm[i].assignment[k] = map_to_node(p);
            }

            swarm[i].current_cost = calculate_cost_host(swarm[i].assignment);

            if (swarm[i].current_cost < swarm[i].pbest_cost) {
                swarm[i].pbest_cost = swarm[i].current_cost;
                memcpy(swarm[i].pbest_pos, swarm[i].pos, sizeof(double) * M);
            }
        }

        for (int i = 0; i < SWARM_SIZE; i++) {
            if (swarm[i].pbest_cost < gbest_cost) {
                gbest_cost = swarm[i].pbest_cost;
                memcpy(gbest_pos, swarm[i].pbest_pos, sizeof(double) * M);
            }
        }
    }

    double end_time = omp_get_wtime();

    for (int k = 0; k < M; k++) best_sol[k] = map_to_node(gbest_pos[k]);
    *best_cost = gbest_cost;

    return end_time - start_time;
}

// =====================================================================================
// 6. PARTICLE SWARM OPTIMIZATION - CUDA C
// =====================================================================================
__global__ void pso_init_swarm_kernel(double *pos, double *vel, double *pbest_pos,
                                      int *assignment, unsigned int *seeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SWARM_SIZE) return;

    unsigned int seed = seeds[i];
    for (int k = 0; k < M; k++) {
        double p = gpu_rand_double(&seed) * (double)P;
        pos[i * M + k] = p;
        vel[i * M + k] = (gpu_rand_double(&seed) * 2.0 - 1.0) * V_MAX * 0.5;
        pbest_pos[i * M + k] = p;
        assignment[i * M + k] = gpu_map_to_node(p);
    }
    seeds[i] = seed;
}

__global__ void pso_update_particles_kernel(double *pos, double *vel,
                                           const double *pbest_pos, const double *gbest_pos,
                                           int *assignment, unsigned int *seeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SWARM_SIZE) return;

    unsigned int seed = seeds[i];

    for (int k = 0; k < M; k++) {
        double r1 = gpu_rand_double(&seed);
        double r2 = gpu_rand_double(&seed);

        double v = W * vel[i * M + k] +
                   C1 * r1 * (pbest_pos[i * M + k] - pos[i * M + k]) +
                   C2 * r2 * (gbest_pos[k] - pos[i * M + k]);

        if (v > V_MAX) v = V_MAX;
        if (v < -V_MAX) v = -V_MAX;
        vel[i * M + k] = v;

        double p = pos[i * M + k] + v;
        if (p < 0.0) p = 0.0;
        if (p >= (double)P) p = (double)P - 1e-4;
        pos[i * M + k] = p;

        assignment[i * M + k] = gpu_map_to_node(p);
    }

    seeds[i] = seed;
}

__global__ void pso_update_bests_kernel(const double *pos, const double *current_costs,
                                       double *pbest_pos, double *pbest_cost,
                                       double *gbest_pos, double *gbest_cost) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int best_p_idx = -1;
        double min_p_cost = *gbest_cost;

        for (int i = 0; i < SWARM_SIZE; i++) {
            if (current_costs[i] < pbest_cost[i]) {
                pbest_cost[i] = current_costs[i];
                for (int k = 0; k < M; k++) {
                    pbest_pos[i * M + k] = pos[i * M + k];
                }
            }
            if (pbest_cost[i] < min_p_cost) {
                min_p_cost = pbest_cost[i];
                best_p_idx = i;
            }
        }

        if (best_p_idx != -1) {
            *gbest_cost = min_p_cost;
            for (int k = 0; k < M; k++) {
                gbest_pos[k] = pbest_pos[best_p_idx * M + k];
            }
        }
    }
}

double run_cuda_pso(int *best_sol, double *best_cost) {
    double *d_pos, *d_vel, *d_pbest_pos;
    double *d_pbest_cost, *d_current_costs, *d_gbest_pos, *d_gbest_cost;
    int *d_assignment;
    unsigned int *d_seeds;

    CHECK_CUDA(cudaMalloc(&d_pos, SWARM_SIZE * M * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vel, SWARM_SIZE * M * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_pbest_pos, SWARM_SIZE * M * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_pbest_cost, SWARM_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_current_costs, SWARM_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_gbest_pos, M * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_gbest_cost, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_assignment, SWARM_SIZE * M * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_seeds, SWARM_SIZE * sizeof(unsigned int)));

    unsigned int h_seeds[SWARM_SIZE];
    for (int i = 0; i < SWARM_SIZE; i++) h_seeds[i] = 42 + i * 29;
    CHECK_CUDA(cudaMemcpy(d_seeds, h_seeds, SWARM_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice));

    double init_gbest = 1e12;
    CHECK_CUDA(cudaMemcpy(d_gbest_cost, &init_gbest, sizeof(double), cudaMemcpyHostToDevice));

    // Initialize pbest_cost with high value
    double init_pbests[SWARM_SIZE];
    for (int i = 0; i < SWARM_SIZE; i++) init_pbests[i] = 1e12;
    CHECK_CUDA(cudaMemcpy(d_pbest_cost, init_pbests, SWARM_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 64;
    int numBlocks = (SWARM_SIZE + blockSize - 1) / blockSize;

    double start_time = omp_get_wtime();

    // 1. Initialize swarm positions & velocities
    pso_init_swarm_kernel<<<numBlocks, blockSize>>>(d_pos, d_vel, d_pbest_pos, d_assignment, d_seeds);

    // 2. Initial fitness evaluation
    evaluate_tap_block_kernel<<<SWARM_SIZE, 128>>>(d_assignment, d_current_costs, d_E, d_C, SWARM_SIZE);

    // 3. Initial pbest & gbest update
    pso_update_bests_kernel<<<1, 1>>>(d_pos, d_current_costs, d_pbest_pos, d_pbest_cost, d_gbest_pos, d_gbest_cost);

    // 4. Optimization iterations
    for (int iter = 0; iter < MAX_ITER; iter++) {
        pso_update_particles_kernel<<<numBlocks, blockSize>>>(d_pos, d_vel, d_pbest_pos, d_gbest_pos,
                                                              d_assignment, d_seeds);
        evaluate_tap_block_kernel<<<SWARM_SIZE, 128>>>(d_assignment, d_current_costs, d_E, d_C, SWARM_SIZE);
        pso_update_bests_kernel<<<1, 1>>>(d_pos, d_current_costs, d_pbest_pos, d_pbest_cost, d_gbest_pos, d_gbest_cost);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double end_time = omp_get_wtime();

    double h_gbest_pos[M];
    CHECK_CUDA(cudaMemcpy(h_gbest_pos, d_gbest_pos, M * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(best_cost, d_gbest_cost, sizeof(double), cudaMemcpyDeviceToHost));

    for (int k = 0; k < M; k++) best_sol[k] = map_to_node(h_gbest_pos[k]);

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_pbest_pos);
    cudaFree(d_pbest_cost);
    cudaFree(d_current_costs);
    cudaFree(d_gbest_pos);
    cudaFree(d_gbest_cost);
    cudaFree(d_assignment);
    cudaFree(d_seeds);

    return end_time - start_time;
}

// =====================================================================================
// MAIN FUNCTION & BENCHMARK COMPARISON
// =====================================================================================
int main(int argc, char **argv) {
    printf("====================================================================================\n");
    printf(" HPC LAB 7: TASK ALLOCATION PROBLEM (TAP) - METAHEURISTICS COMPARISON\n");
    printf(" Problem: Tasks M = %d | Nodes P = %d | Max OpenMP Threads: %d\n", M, P, omp_get_max_threads());
    printf(" GA Config : Population = %d | Generations = %d | Pc = %.2f | Pm = %.2f\n",
           POP_SIZE, MAX_GEN, CROSSOVER_RATE, MUTATION_RATE);
    printf(" PSO Config: Swarm Size = %d | Iterations = %d | w = %.2f | c1 = %.2f | c2 = %.2f\n",
           SWARM_SIZE, MAX_ITER, W, C1, C2);
    printf("====================================================================================\n\n");

    // Initialize problem matrices
    init_problem_data();

    int sol_seq_ga[M], sol_omp_ga[M], sol_cuda_ga[M];
    int sol_seq_pso[M], sol_omp_pso[M], sol_cuda_pso[M];
    double cost_seq_ga, cost_omp_ga, cost_cuda_ga;
    double cost_seq_pso, cost_omp_pso, cost_cuda_pso;

    // --- GENETIC ALGORITHM ---
    printf("[1/6] Running Sequential Genetic Algorithm...\n");
    double t_seq_ga = run_sequential_ga(sol_seq_ga, &cost_seq_ga);
    printf("      -> Completed in %8.4f s | Best Cost: %.2f\n", t_seq_ga, cost_seq_ga);

    printf("[2/6] Running OpenMP Parallel Genetic Algorithm...\n");
    double t_omp_ga = run_openmp_ga(sol_omp_ga, &cost_omp_ga);
    printf("      -> Completed in %8.4f s | Best Cost: %.2f\n", t_omp_ga, cost_omp_ga);

    printf("[3/6] Running CUDA C Parallel Genetic Algorithm...\n");
    double t_cuda_ga = run_cuda_ga(sol_cuda_ga, &cost_cuda_ga);
    printf("      -> Completed in %8.4f s | Best Cost: %.2f\n\n", t_cuda_ga, cost_cuda_ga);

    // --- PARTICLE SWARM OPTIMIZATION ---
    printf("[4/6] Running Sequential Particle Swarm Optimization...\n");
    double t_seq_pso = run_sequential_pso(sol_seq_pso, &cost_seq_pso);
    printf("      -> Completed in %8.4f s | Best Cost: %.2f\n", t_seq_pso, cost_seq_pso);

    printf("[5/6] Running OpenMP Parallel Particle Swarm Optimization...\n");
    double t_omp_pso = run_openmp_pso(sol_omp_pso, &cost_omp_pso);
    printf("      -> Completed in %8.4f s | Best Cost: %.2f\n", t_omp_pso, cost_omp_pso);

    printf("[6/6] Running CUDA C Parallel Particle Swarm Optimization...\n");
    double t_cuda_pso = run_cuda_pso(sol_cuda_pso, &cost_cuda_pso);
    printf("      -> Completed in %8.4f s | Best Cost: %.2f\n\n", t_cuda_pso, cost_cuda_pso);

    // =====================================================================================
    // SUMMARY COMPARISON TABLE
    // =====================================================================================
    printf("====================================================================================\n");
    printf("                     PERFORMANCE AND SPEEDUP SUMMARY (TAP)                          \n");
    printf("====================================================================================\n");
    printf("+--------------------------------------+---------------+------------+------------+\n");
    printf("| Algorithm & Implementation           | Execution (s) | Best Cost  | Speedup    |\n");
    printf("+--------------------------------------+---------------+------------+------------+\n");
    printf("| Genetic Algorithm (Sequential)       | %10.4f s  | %10.2f | %9.2fx |\n", t_seq_ga, cost_seq_ga, 1.00);
    printf("| Genetic Algorithm (OpenMP %2d th)    | %10.4f s  | %10.2f | %9.2fx |\n", omp_get_max_threads(), t_omp_ga, cost_omp_ga, t_seq_ga / t_omp_ga);
    printf("| Genetic Algorithm (CUDA C GPU)       | %10.4f s  | %10.2f | %9.2fx |\n", t_cuda_ga, cost_cuda_ga, t_seq_ga / t_cuda_ga);
    printf("+--------------------------------------+---------------+------------+------------+\n");
    printf("| Particle Swarm (Sequential)          | %10.4f s  | %10.2f | %9.2fx |\n", t_seq_pso, cost_seq_pso, 1.00);
    printf("| Particle Swarm (OpenMP %2d th)       | %10.4f s  | %10.2f | %9.2fx |\n", omp_get_max_threads(), t_omp_pso, cost_omp_pso, t_seq_pso / t_omp_pso);
    printf("| Particle Swarm (CUDA C GPU)          | %10.4f s  | %10.2f | %9.2fx |\n", t_cuda_pso, cost_cuda_pso, t_seq_pso / t_cuda_pso);
    printf("+--------------------------------------+---------------+------------+------------+\n\n");

    // Export CSV results for plotting
    FILE *f = fopen("tap_results.csv", "w");
    if (f) {
        fprintf(f, "Algorithm,Version,Time_Sec,Best_Cost,Speedup\n");
        fprintf(f, "GA,Sequential,%f,%f,%f\n", t_seq_ga, cost_seq_ga, 1.0);
        fprintf(f, "GA,OpenMP,%f,%f,%f\n", t_omp_ga, cost_omp_ga, t_seq_ga / t_omp_ga);
        fprintf(f, "GA,CUDA,%f,%f,%f\n", t_cuda_ga, cost_cuda_ga, t_seq_ga / t_cuda_ga);
        fprintf(f, "PSO,Sequential,%f,%f,%f\n", t_seq_pso, cost_seq_pso, 1.0);
        fprintf(f, "PSO,OpenMP,%f,%f,%f\n", t_omp_pso, cost_omp_pso, t_seq_pso / t_omp_pso);
        fprintf(f, "PSO,CUDA,%f,%f,%f\n", t_cuda_pso, cost_cuda_pso, t_seq_pso / t_cuda_pso);
        fclose(f);
        printf("Results saved to tap_results.csv successfully.\n");
    }

    cleanup_problem_data();
    return 0;
}
