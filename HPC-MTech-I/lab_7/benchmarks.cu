/*
 * =====================================================================================
 * HPC Lab 7: Testing Metaheuristics on 5 Standard Benchmark Functions
 * Comparison of Sequential, OpenMP, and CUDA C Implementations
 *
 * Utilized Metaheuristics:
 *  1. Genetic Algorithm (Real-Coded GA)
 *  2. Particle Swarm Optimization (PSO)
 *
 * Benchmark Functions (D = 30):
 *  1. Sphere Function (Unimodal, Continuous, Convex)
 *  2. Rosenbrock Function (Valley-shaped, Non-convex)
 *  3. Rastrigin Function (Highly Multimodal, Periodic Traps)
 *  4. Ackley Function (Multimodal, Steep Central Basin)
 *  5. Griewank Function (Multimodal, Interdependent Product Component)
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// Benchmark Configuration
#define DIM             30      // Dimension of benchmark problems
#define NUM_FUNCS       5       // Number of benchmark functions
#define POP_SIZE        1024     // Population / Swarm size
#define MAX_ITER        500     // Number of generations / iterations

// PSO Hyperparameters
#define W               0.729   // Inertia weight
#define C1              1.494   // Cognitive parameter
#define C2              1.494   // Social parameter

// GA Hyperparameters
#define CROSSOVER_RATE  0.85
#define MUTATION_RATE   0.05

// Benchmark Function Metadata
typedef struct {
    const char *name;
    const char *type;
    double lower_bound;
    double upper_bound;
    double true_optimum;
} BenchmarkFunc;

static const BenchmarkFunc BENCHMARKS[NUM_FUNCS] = {
    {"Sphere",     "Unimodal, Convex",        -5.12,    5.12,   0.0},
    {"Rosenbrock", "Valley-Shaped Non-Convex", -5.00,   10.00,   0.0},
    {"Rastrigin",  "Highly Multimodal",       -5.12,    5.12,   0.0},
    {"Ackley",      "Multimodal Steep Funnel",-32.768,  32.768, 0.0},
    {"Griewank",   "Multimodal Coupled",     -600.00,  600.00,  0.0}
};

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// =====================================================================================
// HOST BENCHMARK EVALUATORS
// =====================================================================================
double eval_benchmark_host(int func_id, const double *x, int D) {
    if (func_id == 0) { // Sphere
        double sum = 0.0;
        for (int i = 0; i < D; i++) sum += x[i] * x[i];
        return sum;
    } else if (func_id == 1) { // Rosenbrock
        double sum = 0.0;
        for (int i = 0; i < D - 1; i++) {
            double d1 = x[i + 1] - x[i] * x[i];
            double d2 = x[i] - 1.0;
            sum += 100.0 * d1 * d1 + d2 * d2;
        }
        return sum;
    } else if (func_id == 2) { // Rastrigin
        double sum = 10.0 * D;
        for (int i = 0; i < D; i++) {
            sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
        }
        return sum;
    } else if (func_id == 3) { // Ackley
        double sum1 = 0.0, sum2 = 0.0;
        for (int i = 0; i < D; i++) {
            sum1 += x[i] * x[i];
            sum2 += cos(2.0 * M_PI * x[i]);
        }
        return -20.0 * exp(-0.2 * sqrt(sum1 / (double)D)) - exp(sum2 / (double)D) + 20.0 + M_E;
    } else if (func_id == 4) { // Griewank
        double sum = 0.0;
        double prod = 1.0;
        for (int i = 0; i < D; i++) {
            sum += x[i] * x[i];
            prod *= cos(x[i] / sqrt((double)(i + 1)));
        }
        return sum / 4000.0 - prod + 1.0;
    }
    return 0.0;
}

// =====================================================================================
// DEVICE BENCHMARK EVALUATORS
// =====================================================================================
__device__ double eval_benchmark_device(int func_id, const double *x, int D) {
    if (func_id == 0) { // Sphere
        double sum = 0.0;
        for (int i = 0; i < D; i++) sum += x[i] * x[i];
        return sum;
    } else if (func_id == 1) { // Rosenbrock
        double sum = 0.0;
        for (int i = 0; i < D - 1; i++) {
            double d1 = x[i + 1] - x[i] * x[i];
            double d2 = x[i] - 1.0;
            sum += 100.0 * d1 * d1 + d2 * d2;
        }
        return sum;
    } else if (func_id == 2) { // Rastrigin
        double sum = 10.0 * D;
        for (int i = 0; i < D; i++) {
            sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
        }
        return sum;
    } else if (func_id == 3) { // Ackley
        double sum1 = 0.0, sum2 = 0.0;
        for (int i = 0; i < D; i++) {
            sum1 += x[i] * x[i];
            sum2 += cos(2.0 * M_PI * x[i]);
        }
        return -20.0 * exp(-0.2 * sqrt(sum1 / (double)D)) - exp(sum2 / (double)D) + 20.0 + M_E;
    } else if (func_id == 4) { // Griewank
        double sum = 0.0;
        double prod = 1.0;
        for (int i = 0; i < D; i++) {
            sum += x[i] * x[i];
            prod *= cos(x[i] / sqrt((double)(i + 1)));
        }
        return sum / 4000.0 - prod + 1.0;
    }
    return 0.0;
}

// Fast thread-safe PRNGs
static inline double host_rand_double(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (double)((*seed / 65536) % 32768) / 32768.0;
}

static inline int host_rand_int(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (int)((*seed / 65536) % 32768);
}

__device__ inline double gpu_rand_double(unsigned int *seed) {
    *seed = *seed * 1103515245ULL + 12345ULL;
    return (double)((*seed / 65536) % 32768) / 32768.0;
}

__device__ inline unsigned int gpu_rand_int(unsigned int *seed) {
    *seed = *seed * 1103515245ULL + 12345ULL;
    return (*seed / 65536) % 32768;
}

// =====================================================================================
// 1. PSO - SEQUENTIAL
// =====================================================================================
double pso_sequential(int func_id, double *best_val) {
    double lb = BENCHMARKS[func_id].lower_bound;
    double ub = BENCHMARKS[func_id].upper_bound;
    double v_max = 0.2 * (ub - lb);

    double pos[POP_SIZE][DIM];
    double vel[POP_SIZE][DIM];
    double pbest_pos[POP_SIZE][DIM];
    double pbest_val[POP_SIZE];
    double gbest_pos[DIM];
    double gbest_val = 1e20;

    unsigned int seed = 42 + func_id * 101;

    for (int i = 0; i < POP_SIZE; i++) {
        for (int d = 0; d < DIM; d++) {
            pos[i][d] = lb + host_rand_double(&seed) * (ub - lb);
            vel[i][d] = (host_rand_double(&seed) * 2.0 - 1.0) * v_max;
            pbest_pos[i][d] = pos[i][d];
        }
        pbest_val[i] = eval_benchmark_host(func_id, pos[i], DIM);
        if (pbest_val[i] < gbest_val) {
            gbest_val = pbest_val[i];
            memcpy(gbest_pos, pbest_pos[i], sizeof(double) * DIM);
        }
    }

    double t0 = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < POP_SIZE; i++) {
            for (int d = 0; d < DIM; d++) {
                double r1 = host_rand_double(&seed);
                double r2 = host_rand_double(&seed);

                double v = W * vel[i][d] +
                           C1 * r1 * (pbest_pos[i][d] - pos[i][d]) +
                           C2 * r2 * (gbest_pos[d] - pos[i][d]);

                if (v > v_max) v = v_max;
                if (v < -v_max) v = -v_max;
                vel[i][d] = v;

                double p = pos[i][d] + v;
                if (p < lb) p = lb;
                if (p > ub) p = ub;
                pos[i][d] = p;
            }

            double val = eval_benchmark_host(func_id, pos[i], DIM);
            if (val < pbest_val[i]) {
                pbest_val[i] = val;
                memcpy(pbest_pos[i], pos[i], sizeof(double) * DIM);
                if (val < gbest_val) {
                    gbest_val = val;
                    memcpy(gbest_pos, pos[i], sizeof(double) * DIM);
                }
            }
        }
    }

    double t1 = omp_get_wtime();
    *best_val = gbest_val;
    return (t1 - t0);
}

// =====================================================================================
// 2. PSO - OPENMP
// =====================================================================================
double pso_openmp(int func_id, double *best_val) {
    double lb = BENCHMARKS[func_id].lower_bound;
    double ub = BENCHMARKS[func_id].upper_bound;
    double v_max = 0.2 * (ub - lb);

    double pos[POP_SIZE][DIM];
    double vel[POP_SIZE][DIM];
    double pbest_pos[POP_SIZE][DIM];
    double pbest_val[POP_SIZE];
    double gbest_pos[DIM];
    double gbest_val = 1e20;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < POP_SIZE; i++) {
        unsigned int seed = 42 + func_id * 101 + i * 31;
        for (int d = 0; d < DIM; d++) {
            pos[i][d] = lb + host_rand_double(&seed) * (ub - lb);
            vel[i][d] = (host_rand_double(&seed) * 2.0 - 1.0) * v_max;
            pbest_pos[i][d] = pos[i][d];
        }
        pbest_val[i] = eval_benchmark_host(func_id, pos[i], DIM);
    }

    for (int i = 0; i < POP_SIZE; i++) {
        if (pbest_val[i] < gbest_val) {
            gbest_val = pbest_val[i];
            memcpy(gbest_pos, pbest_pos[i], sizeof(double) * DIM);
        }
    }

    double t0 = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < POP_SIZE; i++) {
            unsigned int seed = iter * 1000 + i * 23 + omp_get_thread_num();
            for (int d = 0; d < DIM; d++) {
                double r1 = host_rand_double(&seed);
                double r2 = host_rand_double(&seed);

                double v = W * vel[i][d] +
                           C1 * r1 * (pbest_pos[i][d] - pos[i][d]) +
                           C2 * r2 * (gbest_pos[d] - pos[i][d]);

                if (v > v_max) v = v_max;
                if (v < -v_max) v = -v_max;
                vel[i][d] = v;

                double p = pos[i][d] + v;
                if (p < lb) p = lb;
                if (p > ub) p = ub;
                pos[i][d] = p;
            }

            double val = eval_benchmark_host(func_id, pos[i], DIM);
            if (val < pbest_val[i]) {
                pbest_val[i] = val;
                memcpy(pbest_pos[i], pos[i], sizeof(double) * DIM);
            }
        }

        for (int i = 0; i < POP_SIZE; i++) {
            if (pbest_val[i] < gbest_val) {
                gbest_val = pbest_val[i];
                memcpy(gbest_pos, pbest_pos[i], sizeof(double) * DIM);
            }
        }
    }

    double t1 = omp_get_wtime();
    *best_val = gbest_val;
    return (t1 - t0);
}

// =====================================================================================
// 3. PSO - CUDA C
// =====================================================================================
__global__ void pso_bench_init_kernel(int func_id, double *pos, double *vel, double *pbest_pos,
                                      double *pbest_val, double lb, double ub, double v_max,
                                      unsigned int *seeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POP_SIZE) return;

    unsigned int seed = seeds[i];
    for (int d = 0; d < DIM; d++) {
        double p = lb + gpu_rand_double(&seed) * (ub - lb);
        pos[i * DIM + d] = p;
        vel[i * DIM + d] = (gpu_rand_double(&seed) * 2.0 - 1.0) * v_max;
        pbest_pos[i * DIM + d] = p;
    }
    pbest_val[i] = eval_benchmark_device(func_id, &pos[i * DIM], DIM);
    seeds[i] = seed;
}

__global__ void pso_bench_update_kernel(int func_id, double *pos, double *vel,
                                        double *pbest_pos, double *pbest_val,
                                        const double *gbest_pos,
                                        double lb, double ub, double v_max,
                                        unsigned int *seeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POP_SIZE) return;

    unsigned int seed = seeds[i];

    for (int d = 0; d < DIM; d++) {
        double r1 = gpu_rand_double(&seed);
        double r2 = gpu_rand_double(&seed);

        double v = W * vel[i * DIM + d] +
                   C1 * r1 * (pbest_pos[i * DIM + d] - pos[i * DIM + d]) +
                   C2 * r2 * (gbest_pos[d] - pos[i * DIM + d]);

        if (v > v_max) v = v_max;
        if (v < -v_max) v = -v_max;
        vel[i * DIM + d] = v;

        double p = pos[i * DIM + d] + v;
        if (p < lb) p = lb;
        if (p > ub) p = ub;
        pos[i * DIM + d] = p;
    }

    double val = eval_benchmark_device(func_id, &pos[i * DIM], DIM);
    if (val < pbest_val[i]) {
        pbest_val[i] = val;
        for (int d = 0; d < DIM; d++) {
            pbest_pos[i * DIM + d] = pos[i * DIM + d];
        }
    }

    seeds[i] = seed;
}

__global__ void pso_bench_gbest_kernel(const double *pbest_pos, const double *pbest_val,
                                       double *gbest_pos, double *gbest_val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int best_idx = -1;
        double min_val = *gbest_val;

        for (int i = 0; i < POP_SIZE; i++) {
            if (pbest_val[i] < min_val) {
                min_val = pbest_val[i];
                best_idx = i;
            }
        }

        if (best_idx != -1) {
            *gbest_val = min_val;
            for (int d = 0; d < DIM; d++) {
                gbest_pos[d] = pbest_pos[best_idx * DIM + d];
            }
        }
    }
}

double pso_cuda(int func_id, double *best_val) {
    double lb = BENCHMARKS[func_id].lower_bound;
    double ub = BENCHMARKS[func_id].upper_bound;
    double v_max = 0.2 * (ub - lb);

    double *d_pos, *d_vel, *d_pbest_pos, *d_pbest_val;
    double *d_gbest_pos, *d_gbest_val;
    unsigned int *d_seeds;

    CHECK_CUDA(cudaMalloc(&d_pos, POP_SIZE * DIM * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_vel, POP_SIZE * DIM * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_pbest_pos, POP_SIZE * DIM * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_pbest_val, POP_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_gbest_pos, DIM * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_gbest_val, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_seeds, POP_SIZE * sizeof(unsigned int)));

    unsigned int h_seeds[POP_SIZE];
    for (int i = 0; i < POP_SIZE; i++) h_seeds[i] = 42 + func_id * 101 + i * 31;
    CHECK_CUDA(cudaMemcpy(d_seeds, h_seeds, POP_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice));

    double init_gbest = 1e20;
    CHECK_CUDA(cudaMemcpy(d_gbest_val, &init_gbest, sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 128;
    int numBlocks = (POP_SIZE + blockSize - 1) / blockSize;

    double t0 = omp_get_wtime();

    pso_bench_init_kernel<<<numBlocks, blockSize>>>(func_id, d_pos, d_vel, d_pbest_pos,
                                                    d_pbest_val, lb, ub, v_max, d_seeds);
    pso_bench_gbest_kernel<<<1, 1>>>(d_pbest_pos, d_pbest_val, d_gbest_pos, d_gbest_val);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        pso_bench_update_kernel<<<numBlocks, blockSize>>>(func_id, d_pos, d_vel,
                                                          d_pbest_pos, d_pbest_val,
                                                          d_gbest_pos, lb, ub, v_max, d_seeds);
        pso_bench_gbest_kernel<<<1, 1>>>(d_pbest_pos, d_pbest_val, d_gbest_pos, d_gbest_val);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double t1 = omp_get_wtime();

    CHECK_CUDA(cudaMemcpy(best_val, d_gbest_val, sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_pbest_pos);
    cudaFree(d_pbest_val);
    cudaFree(d_gbest_pos);
    cudaFree(d_gbest_val);
    cudaFree(d_seeds);

    return (t1 - t0);
}

// =====================================================================================
// 4. REAL-CODED GA - SEQUENTIAL
// =====================================================================================
double ga_sequential(int func_id, double *best_val) {
    double lb = BENCHMARKS[func_id].lower_bound;
    double ub = BENCHMARKS[func_id].upper_bound;

    double pop[POP_SIZE][DIM];
    double next_pop[POP_SIZE][DIM];
    double costs[POP_SIZE];
    double next_costs[POP_SIZE];

    unsigned int seed = 42 + func_id * 73;

    for (int i = 0; i < POP_SIZE; i++) {
        for (int d = 0; d < DIM; d++) {
            pop[i][d] = lb + host_rand_double(&seed) * (ub - lb);
        }
        costs[i] = eval_benchmark_host(func_id, pop[i], DIM);
    }

    double t0 = omp_get_wtime();

    for (int gen = 0; gen < MAX_ITER; gen++) {
        // Find 2 elites
        int e1 = 0, e2 = 1;
        if (costs[e2] < costs[e1]) { e1 = 1; e2 = 0; }
        for (int i = 2; i < POP_SIZE; i++) {
            if (costs[i] < costs[e1]) { e2 = e1; e1 = i; }
            else if (costs[i] < costs[e2]) { e2 = i; }
        }

        // Elitism
        memcpy(next_pop[0], pop[e1], sizeof(double) * DIM);
        next_costs[0] = costs[e1];
        memcpy(next_pop[1], pop[e2], sizeof(double) * DIM);
        next_costs[1] = costs[e2];

        // Reproduction
        for (int i = 2; i < POP_SIZE; i++) {
            int p1 = host_rand_int(&seed) % POP_SIZE;
            for (int t = 1; t < 3; t++) {
                int cand = host_rand_int(&seed) % POP_SIZE;
                if (costs[cand] < costs[p1]) p1 = cand;
            }

            int p2 = host_rand_int(&seed) % POP_SIZE;
            for (int t = 1; t < 3; t++) {
                int cand = host_rand_int(&seed) % POP_SIZE;
                if (costs[cand] < costs[p2]) p2 = cand;
            }

            // Arithmetic crossover
            if (host_rand_double(&seed) < CROSSOVER_RATE) {
                double alpha = host_rand_double(&seed);
                for (int d = 0; d < DIM; d++) {
                    next_pop[i][d] = alpha * pop[p1][d] + (1.0 - alpha) * pop[p2][d];
                }
            } else {
                memcpy(next_pop[i], pop[p1], sizeof(double) * DIM);
            }

            // Mutation
            for (int d = 0; d < DIM; d++) {
                if (host_rand_double(&seed) < MUTATION_RATE) {
                    double delta = (host_rand_double(&seed) * 2.0 - 1.0) * 0.1 * (ub - lb);
                    double p = next_pop[i][d] + delta;
                    if (p < lb) p = lb;
                    if (p > ub) p = ub;
                    next_pop[i][d] = p;
                }
            }

            next_costs[i] = eval_benchmark_host(func_id, next_pop[i], DIM);
        }

        memcpy(pop, next_pop, sizeof(double) * POP_SIZE * DIM);
        memcpy(costs, next_costs, sizeof(double) * POP_SIZE);
    }

    double t1 = omp_get_wtime();

    double min_cost = costs[0];
    for (int i = 1; i < POP_SIZE; i++) {
        if (costs[i] < min_cost) min_cost = costs[i];
    }
    *best_val = min_cost;

    return (t1 - t0);
}

// =====================================================================================
// 5. REAL-CODED GA - OPENMP
// =====================================================================================
double ga_openmp(int func_id, double *best_val) {
    double lb = BENCHMARKS[func_id].lower_bound;
    double ub = BENCHMARKS[func_id].upper_bound;

    double pop[POP_SIZE][DIM];
    double next_pop[POP_SIZE][DIM];
    double costs[POP_SIZE];
    double next_costs[POP_SIZE];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < POP_SIZE; i++) {
        unsigned int seed = 42 + func_id * 73 + i * 29;
        for (int d = 0; d < DIM; d++) {
            pop[i][d] = lb + host_rand_double(&seed) * (ub - lb);
        }
        costs[i] = eval_benchmark_host(func_id, pop[i], DIM);
    }

    double t0 = omp_get_wtime();

    for (int gen = 0; gen < MAX_ITER; gen++) {
        int e1 = 0, e2 = 1;
        if (costs[e2] < costs[e1]) { e1 = 1; e2 = 0; }
        for (int i = 2; i < POP_SIZE; i++) {
            if (costs[i] < costs[e1]) { e2 = e1; e1 = i; }
            else if (costs[i] < costs[e2]) { e2 = i; }
        }

        memcpy(next_pop[0], pop[e1], sizeof(double) * DIM);
        next_costs[0] = costs[e1];
        memcpy(next_pop[1], pop[e2], sizeof(double) * DIM);
        next_costs[1] = costs[e2];

        #pragma omp parallel for schedule(static)
        for (int i = 2; i < POP_SIZE; i++) {
            unsigned int seed = gen * 1000 + i * 17 + omp_get_thread_num();

            int p1 = host_rand_int(&seed) % POP_SIZE;
            for (int t = 1; t < 3; t++) {
                int cand = host_rand_int(&seed) % POP_SIZE;
                if (costs[cand] < costs[p1]) p1 = cand;
            }

            int p2 = host_rand_int(&seed) % POP_SIZE;
            for (int t = 1; t < 3; t++) {
                int cand = host_rand_int(&seed) % POP_SIZE;
                if (costs[cand] < costs[p2]) p2 = cand;
            }

            if (host_rand_double(&seed) < CROSSOVER_RATE) {
                double alpha = host_rand_double(&seed);
                for (int d = 0; d < DIM; d++) {
                    next_pop[i][d] = alpha * pop[p1][d] + (1.0 - alpha) * pop[p2][d];
                }
            } else {
                for (int d = 0; d < DIM; d++) next_pop[i][d] = pop[p1][d];
            }

            for (int d = 0; d < DIM; d++) {
                if (host_rand_double(&seed) < MUTATION_RATE) {
                    double delta = (host_rand_double(&seed) * 2.0 - 1.0) * 0.1 * (ub - lb);
                    double p = next_pop[i][d] + delta;
                    if (p < lb) p = lb;
                    if (p > ub) p = ub;
                    next_pop[i][d] = p;
                }
            }

            next_costs[i] = eval_benchmark_host(func_id, next_pop[i], DIM);
        }

        memcpy(pop, next_pop, sizeof(double) * POP_SIZE * DIM);
        memcpy(costs, next_costs, sizeof(double) * POP_SIZE);
    }

    double t1 = omp_get_wtime();

    double min_cost = costs[0];
    for (int i = 1; i < POP_SIZE; i++) {
        if (costs[i] < min_cost) min_cost = costs[i];
    }
    *best_val = min_cost;

    return (t1 - t0);
}

// =====================================================================================
// 6. REAL-CODED GA - CUDA C
// =====================================================================================
__global__ void ga_bench_init_kernel(int func_id, double *pop, double *costs,
                                     double lb, double ub, unsigned int *seeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POP_SIZE) return;

    unsigned int seed = seeds[i];
    for (int d = 0; d < DIM; d++) {
        pop[i * DIM + d] = lb + gpu_rand_double(&seed) * (ub - lb);
    }
    costs[i] = eval_benchmark_device(func_id, &pop[i * DIM], DIM);
    seeds[i] = seed;
}

__global__ void ga_bench_find_elites_kernel(const double *costs, int *elite_indices) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int e1 = 0, e2 = 1;
        if (costs[e2] < costs[e1]) { e1 = 1; e2 = 0; }
        for (int i = 2; i < POP_SIZE; i++) {
            if (costs[i] < costs[e1]) { e2 = e1; e1 = i; }
            else if (costs[i] < costs[e2]) { e2 = i; }
        }
        elite_indices[0] = e1;
        elite_indices[1] = e2;
    }
}

__global__ void ga_bench_reproduce_kernel(int func_id, const double *pop, const double *costs,
                                          double *next_pop, double *next_costs,
                                          const int *elite_indices, double lb, double ub,
                                          unsigned int *seeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POP_SIZE) return;

    // Preserved elites
    if (i == 0 || i == 1) {
        int elite = elite_indices[i];
        next_costs[i] = costs[elite];
        for (int d = 0; d < DIM; d++) {
            next_pop[i * DIM + d] = pop[elite * DIM + d];
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

    // Arithmetic crossover
    if (gpu_rand_double(&seed) < CROSSOVER_RATE) {
        double alpha = gpu_rand_double(&seed);
        for (int d = 0; d < DIM; d++) {
            next_pop[i * DIM + d] = alpha * pop[p1 * DIM + d] + (1.0 - alpha) * pop[p2 * DIM + d];
        }
    } else {
        for (int d = 0; d < DIM; d++) {
            next_pop[i * DIM + d] = pop[p1 * DIM + d];
        }
    }

    // Mutation
    for (int d = 0; d < DIM; d++) {
        if (gpu_rand_double(&seed) < MUTATION_RATE) {
            double delta = (gpu_rand_double(&seed) * 2.0 - 1.0) * 0.1 * (ub - lb);
            double p = next_pop[i * DIM + d] + delta;
            if (p < lb) p = lb;
            if (p > ub) p = ub;
            next_pop[i * DIM + d] = p;
        }
    }

    next_costs[i] = eval_benchmark_device(func_id, &next_pop[i * DIM], DIM);
    seeds[i] = seed;
}

double ga_cuda(int func_id, double *best_val) {
    double lb = BENCHMARKS[func_id].lower_bound;
    double ub = BENCHMARKS[func_id].upper_bound;

    double *d_pop, *d_next_pop;
    double *d_costs, *d_next_costs;
    int *d_elite_indices;
    unsigned int *d_seeds;

    CHECK_CUDA(cudaMalloc(&d_pop, POP_SIZE * DIM * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_next_pop, POP_SIZE * DIM * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_costs, POP_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_next_costs, POP_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_elite_indices, 2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_seeds, POP_SIZE * sizeof(unsigned int)));

    unsigned int h_seeds[POP_SIZE];
    for (int i = 0; i < POP_SIZE; i++) h_seeds[i] = 42 + func_id * 73 + i * 29;
    CHECK_CUDA(cudaMemcpy(d_seeds, h_seeds, POP_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice));

    int blockSize = 128;
    int numBlocks = (POP_SIZE + blockSize - 1) / blockSize;

    double t0 = omp_get_wtime();

    ga_bench_init_kernel<<<numBlocks, blockSize>>>(func_id, d_pop, d_costs, lb, ub, d_seeds);

    for (int gen = 0; gen < MAX_ITER; gen++) {
        ga_bench_find_elites_kernel<<<1, 1>>>(d_costs, d_elite_indices);
        ga_bench_reproduce_kernel<<<numBlocks, blockSize>>>(func_id, d_pop, d_costs,
                                                            d_next_pop, d_next_costs,
                                                            d_elite_indices, lb, ub, d_seeds);

        double *tmp_pop = d_pop; d_pop = d_next_pop; d_next_pop = tmp_pop;
        double *tmp_costs = d_costs; d_costs = d_next_costs; d_next_costs = tmp_costs;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double t1 = omp_get_wtime();

    ga_bench_find_elites_kernel<<<1, 1>>>(d_costs, d_elite_indices);
    int h_elites[2];
    CHECK_CUDA(cudaMemcpy(h_elites, d_elite_indices, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(best_val, &d_costs[h_elites[0]], sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_pop);
    cudaFree(d_next_pop);
    cudaFree(d_costs);
    cudaFree(d_next_costs);
    cudaFree(d_elite_indices);
    cudaFree(d_seeds);

    return (t1 - t0);
}

// =====================================================================================
// MAIN BENCHMARK RUNNER
// =====================================================================================
int main(int argc, char **argv) {
    cudaFree(0); // Warmup CUDA runtime driver context

    printf("=====================================================================================================\n");
    printf(" HPC LAB 7: BENCHMARK FUNCTIONS EVALUATION (GA & PSO: SEQUENTIAL vs OPENMP vs CUDA C)\n");
    printf(" Problem Dimension D = %d | Swarm/Pop Size = %d | Iterations = %d | Max OpenMP Threads: %d\n",
           DIM, POP_SIZE, MAX_ITER, omp_get_max_threads());
    printf(" Benchmark Functions: Sphere, Rosenbrock, Rastrigin, Ackley, Griewank\n");
    printf("=====================================================================================================\n\n");

    FILE *f_csv = fopen("benchmark_results.csv", "w");
    if (f_csv) {
        fprintf(f_csv, "Function,Algorithm,Sequential_Time,OpenMP_Time,CUDA_Time,Speedup_OpenMP,Speedup_CUDA,Best_Seq,Best_OpenMP,Best_CUDA\n");
    }

    for (int f = 0; f < NUM_FUNCS; f++) {
        printf(">>> [Function %d/%d]: %-12s (Domain: [%.2f, %.2f], Optimum = %.2f)\n",
               f + 1, NUM_FUNCS, BENCHMARKS[f].name,
               BENCHMARKS[f].lower_bound, BENCHMARKS[f].upper_bound, BENCHMARKS[f].true_optimum);

        double pso_best_seq, pso_best_omp, pso_best_cuda;
        double ga_best_seq, ga_best_omp, ga_best_cuda;

        // Run PSO
        double t_pso_seq = pso_sequential(f, &pso_best_seq);
        double t_pso_omp = pso_openmp(f, &pso_best_omp);
        double t_pso_cuda = pso_cuda(f, &pso_best_cuda);

        // Run GA
        double t_ga_seq = ga_sequential(f, &ga_best_seq);
        double t_ga_omp = ga_openmp(f, &ga_best_omp);
        double t_ga_cuda = ga_cuda(f, &ga_best_cuda);

        printf("+-----------+---------------+---------------+---------------+------------+------------+----------------+\n");
        printf("| Algorithm | Sequential(s) | OpenMP(s)     | CUDA(s)       | Spdup(OMP) | Spdup(CUDA)| Best Fitness   |\n");
        printf("+-----------+---------------+---------------+---------------+------------+------------+----------------+\n");
        printf("| PSO       | %11.4f s | %11.4f s | %11.4f s | %9.2fx  | %9.2fx  | %14.6e |\n",
               t_pso_seq, t_pso_omp, t_pso_cuda, t_pso_seq / t_pso_omp, t_pso_seq / t_pso_cuda, pso_best_cuda);
        printf("| GA        | %11.4f s | %11.4f s | %11.4f s | %9.2fx  | %9.2fx  | %14.6e |\n",
               t_ga_seq, t_ga_omp, t_ga_cuda, t_ga_seq / t_ga_omp, t_ga_seq / t_ga_cuda, ga_best_cuda);
        printf("+-----------+---------------+---------------+---------------+------------+------------+----------------+\n\n");

        if (f_csv) {
            fprintf(f_csv, "%s,PSO,%f,%f,%f,%f,%f,%e,%e,%e\n",
                    BENCHMARKS[f].name, t_pso_seq, t_pso_omp, t_pso_cuda,
                    t_pso_seq / t_pso_omp, t_pso_seq / t_pso_cuda,
                    pso_best_seq, pso_best_omp, pso_best_cuda);
            fprintf(f_csv, "%s,GA,%f,%f,%f,%f,%f,%e,%e,%e\n",
                    BENCHMARKS[f].name, t_ga_seq, t_ga_omp, t_ga_cuda,
                    t_ga_seq / t_ga_omp, t_ga_seq / t_ga_cuda,
                    ga_best_seq, ga_best_omp, ga_best_cuda);
        }
    }

    if (f_csv) {
        fclose(f_csv);
        printf("Benchmark summary successfully saved to benchmark_results.csv.\n");
    }

    return 0;
}
