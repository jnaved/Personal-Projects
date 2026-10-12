/*
 * HPC Lab 6: Parallel Linear Regression with Two Features
 * Comparison of Sequential, OpenMP, and CUDA C implementations
 * Model: y = w0 + w1*x1 + w2*x2
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define N           10000000    // 10 Million datapoints
#define EPOCHS      100         // Number of gradient descent iterations
#define ALPHA       0.2f        // Learning rate
#define BLOCK_SIZE  256         // Threads per CUDA block

// Ground-truth model parameters
#define TRUE_W0     3.50f
#define TRUE_W1     1.80f
#define TRUE_W2    -2.40f

// Generate synthetic dataset with noise
void generate_data(float *x1, float *x2, float *y, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        x1[i] = -1.0f + 2.0f * ((float)rand() / (float)RAND_MAX);
        x2[i] = -1.0f + 2.0f * ((float)rand() / (float)RAND_MAX);
        float noise = 0.05f * (-1.0f + 2.0f * ((float)rand() / (float)RAND_MAX));
        y[i] = TRUE_W0 + TRUE_W1 * x1[i] + TRUE_W2 * x2[i] + noise;
    }
}

// Compute Mean Squared Error (MSE)
float compute_mse(const float *x1, const float *x2, const float *y,
                  float w0, float w1, float w2, int n) {
    double total_err = 0.0;
    for (int i = 0; i < n; i++) {
        float err = (w0 + w1 * x1[i] + w2 * x2[i]) - y[i];
        total_err += (double)(err * err);
    }
    return (float)(total_err / (2.0 * n));
}

// =====================================================================
// 1. SEQUENTIAL IMPLEMENTATION
// =====================================================================
double run_sequential(const float *x1, const float *x2, const float *y, int n,
                      int epochs, float alpha, float *w0, float *w1, float *w2) {
    *w0 = 0.0f; *w1 = 0.0f; *w2 = 0.0f;
    double t_start = omp_get_wtime();

    for (int ep = 0; ep < epochs; ep++) {
        double g0 = 0.0, g1 = 0.0, g2 = 0.0;
        for (int i = 0; i < n; i++) {
            float err = (*w0 + *w1 * x1[i] + *w2 * x2[i]) - y[i];
            g0 += err;
            g1 += err * x1[i];
            g2 += err * x2[i];
        }
        *w0 -= alpha * (float)(g0 / n);
        *w1 -= alpha * (float)(g1 / n);
        *w2 -= alpha * (float)(g2 / n);
    }

    return (omp_get_wtime() - t_start) * 1000.0; // Return time in ms
}

// =====================================================================
// 2. OPENMP IMPLEMENTATION
// =====================================================================
double run_openmp(const float *x1, const float *x2, const float *y, int n,
                  int epochs, float alpha, float *w0, float *w1, float *w2) {
    *w0 = 0.0f; *w1 = 0.0f; *w2 = 0.0f;
    double t_start = omp_get_wtime();

    for (int ep = 0; ep < epochs; ep++) {
        double g0 = 0.0, g1 = 0.0, g2 = 0.0;
        #pragma omp parallel for reduction(+:g0, g1, g2) schedule(static)
        for (int i = 0; i < n; i++) {
            float err = (*w0 + *w1 * x1[i] + *w2 * x2[i]) - y[i];
            g0 += err;
            g1 += err * x1[i];
            g2 += err * x2[i];
        }
        *w0 -= alpha * (float)(g0 / n);
        *w1 -= alpha * (float)(g1 / n);
        *w2 -= alpha * (float)(g2 / n);
    }

    return (omp_get_wtime() - t_start) * 1000.0; // Return time in ms
}

// =====================================================================
// 3. CUDA IMPLEMENTATION
// =====================================================================
__global__ void compute_gradients_cuda(const float *x1, const float *x2, const float *y,
                                       float *grad, float w0, float w1, float w2, int n) {
    __shared__ float s_g0[BLOCK_SIZE];
    __shared__ float s_g1[BLOCK_SIZE];
    __shared__ float s_g2[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_g0 = 0.0f, local_g1 = 0.0f, local_g2 = 0.0f;
    for (int i = idx; i < n; i += stride) {
        float err = (w0 + w1 * x1[i] + w2 * x2[i]) - y[i];
        local_g0 += err;
        local_g1 += err * x1[i];
        local_g2 += err * x2[i];
    }

    s_g0[tid] = local_g0;
    s_g1[tid] = local_g1;
    s_g2[tid] = local_g2;
    __syncthreads();

    // Block-level tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_g0[tid] += s_g0[tid + s];
            s_g1[tid] += s_g1[tid + s];
            s_g2[tid] += s_g2[tid + s];
        }
        __syncthreads();
    }

    // Block leader adds to global accumulator
    if (tid == 0) {
        atomicAdd(&grad[0], s_g0[0]);
        atomicAdd(&grad[1], s_g1[0]);
        atomicAdd(&grad[2], s_g2[0]);
    }
}

double run_cuda(const float *h_x1, const float *h_x2, const float *h_y, int n,
                int epochs, float alpha, float *w0, float *w1, float *w2) {
    *w0 = 0.0f; *w1 = 0.0f; *w2 = 0.0f;
    size_t bytes = n * sizeof(float);

    float *d_x1, *d_x2, *d_y, *d_grad;
    cudaMalloc(&d_x1, bytes);
    cudaMalloc(&d_x2, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_grad, 3 * sizeof(float));

    double t_start = omp_get_wtime();

    cudaMemcpy(d_x1, h_x1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    int num_blocks = 512;
    float h_grad[3];

    for (int ep = 0; ep < epochs; ep++) {
        cudaMemset(d_grad, 0, 3 * sizeof(float));
        compute_gradients_cuda<<<num_blocks, BLOCK_SIZE>>>(d_x1, d_x2, d_y, d_grad, *w0, *w1, *w2, n);
        cudaMemcpy(h_grad, d_grad, 3 * sizeof(float), cudaMemcpyDeviceToHost);

        *w0 -= alpha * (h_grad[0] / n);
        *w1 -= alpha * (h_grad[1] / n);
        *w2 -= alpha * (h_grad[2] / n);
    }

    double t_end = omp_get_wtime();

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_y);
    cudaFree(d_grad);

    return (t_end - t_start) * 1000.0; // Return total time in ms
}

// =====================================================================
// MAIN BENCHMARK
// =====================================================================
int main() {
    printf("=====================================================================\n");
    printf("  Linear Regression with Two Features (CUDA vs OpenMP vs Sequential)\n");
    printf("  Datapoints N = %'d | Epochs = %d | Learning Rate = %.2f\n", N, EPOCHS, ALPHA);
    printf("  Target Model : y = %.2f + %.2f*x1 + (%.2f)*x2\n", TRUE_W0, TRUE_W1, TRUE_W2);
    printf("=====================================================================\n\n");

    // Allocate memory on host
    float *x1 = (float *)malloc(N * sizeof(float));
    float *x2 = (float *)malloc(N * sizeof(float));
    float *y  = (float *)malloc(N * sizeof(float));
    if (!x1 || !x2 || !y) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    printf("Generating %'d data points...\n", N);
    generate_data(x1, x2, y, N);

    float w0_seq, w1_seq, w2_seq;
    float w0_omp, w1_omp, w2_omp;
    float w0_cu,  w1_cu,  w2_cu;

    // 1. Sequential
    printf("Running Sequential baseline... ");
    fflush(stdout);
    double t_seq = run_sequential(x1, x2, y, N, EPOCHS, ALPHA, &w0_seq, &w1_seq, &w2_seq);
    float mse_seq = compute_mse(x1, x2, y, w0_seq, w1_seq, w2_seq, N);
    printf("Done in %.2f ms\n", t_seq);

    // 2. OpenMP
    printf("Running OpenMP (%d threads)... ", omp_get_max_threads());
    fflush(stdout);
    double t_omp = run_openmp(x1, x2, y, N, EPOCHS, ALPHA, &w0_omp, &w1_omp, &w2_omp);
    float mse_omp = compute_mse(x1, x2, y, w0_omp, w1_omp, w2_omp, N);
    printf("Done in %.2f ms\n", t_omp);

    // 3. CUDA
    printf("Running CUDA on GPU... ");
    fflush(stdout);
    double t_cu = run_cuda(x1, x2, y, N, EPOCHS, ALPHA, &w0_cu, &w1_cu, &w2_cu);
    float mse_cu = compute_mse(x1, x2, y, w0_cu, w1_cu, w2_cu, N);
    printf("Done in %.2f ms\n", t_cu);

    // Results Comparison
    printf("\n=====================================================================\n");
    printf("                      PERFORMANCE COMPARISON                         \n");
    printf("=====================================================================\n");
    printf("Implementation  | Time (ms)  | Speedup  | Final Weights [w0, w1, w2]  | Final MSE\n");
    printf("----------------+------------+----------+-----------------------------+----------\n");
    printf("Sequential      | %9.2f  |   1.00x  | [%.4f, %.4f, %.4f]  | %.6f\n",
           t_seq, w0_seq, w1_seq, w2_seq, mse_seq);
    printf("OpenMP (%2d th) | %9.2f  |  %5.2fx  | [%.4f, %.4f, %.4f]  | %.6f\n",
           omp_get_max_threads(), t_omp, t_seq / t_omp, w0_omp, w1_omp, w2_omp, mse_omp);
    printf("CUDA (GPU)      | %9.2f  |  %5.2fx  | [%.4f, %.4f, %.4f]  | %.6f\n",
           t_cu, t_seq / t_cu, w0_cu, w1_cu, w2_cu, mse_cu);
    printf("=====================================================================\n");

    free(x1);
    free(x2);
    free(y);
    return 0;
}
