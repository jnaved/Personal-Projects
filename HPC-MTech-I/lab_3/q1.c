/*
 * High Performance Computing - Lab Assignment 3
 * Question 1: Parallel Linear Regression using OpenMP and Sequential Comparison
 * 
 * Problem Statement:
 * Implement a parallel version of linear regression using OpenMP.
 * Consider X values are two-dimensional and Y label is one-dimensional,
 * i.e., overall dataspace is 3-dimensional (X1, X2, Y).
 * Apply parallelization on Gradient Descent computation and compare its
 * running time performance using OpenMP with its sequential counterpart.
 * Use a significantly large number of datapoints for performance comparison.
 * 
 * Compilation & Execution:
 *   gcc -O3 -fopenmp lab_3/q1.c -o lab_3/q1.exe -lm
 *   .\lab_3\q1.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// ==========================================
// CONFIGURATION PARAMETERS
// ==========================================
#define N 2000000              // Number of data points (2 Million)
#define NUM_EPOCHS 100         // Number of gradient descent iterations
#define LEARNING_RATE 0.1f     // Learning rate (alpha / eta)
#define NUM_THREADS 8          // Number of OpenMP threads for parallel execution

// Ground truth parameters for synthetic dataset: Y = W1_TRUE * X1 + W2_TRUE * X2 + B_TRUE + noise
#define W1_TRUE 3.50f
#define W2_TRUE 1.50f
#define B_TRUE  2.00f

// ==========================================
// DATA GENERATION & LOSS FUNCTION
// ==========================================
void generate_dataset(float *x1, float *x2, float *y, int n) {
    srand(42); // Fixed seed for reproducible benchmarks
    for (int i = 0; i < n; i++) {
        // Normalized feature coordinates in [0, 1]
        x1[i] = (float)rand() / (float)RAND_MAX;
        x2[i] = (float)rand() / (float)RAND_MAX;

        // Small noise in [-0.05, 0.05]
        float noise = (((float)rand() / (float)RAND_MAX) - 0.5f) * 0.1f;
        y[i] = (W1_TRUE * x1[i]) + (W2_TRUE * x2[i]) + B_TRUE + noise;
    }
}

// Compute Mean Squared Error (MSE) loss: J(w1, w2, b) = 1/(2N) * sum((y_pred - y_actual)^2)
float compute_mse_loss(const float *x1, const float *x2, const float *y, int n, float w1, float w2, float b) {
    double total_error = 0.0;
    #pragma omp parallel for reduction(+:total_error)
    for (int i = 0; i < n; i++) {
        float y_pred = (w1 * x1[i]) + (w2 * x2[i]) + b;
        float err = y_pred - y[i];
        total_error += (double)(err * err);
    }
    return (float)(total_error / (2.0 * (double)n));
}

// ==========================================
// 1. SEQUENTIAL GRADIENT DESCENT
// ==========================================
double linear_regression_sequential(const float *x1, const float *x2, const float *y, int n, int epochs, float alpha, float *final_w1, float *final_w2, float *final_b) {
    float w1 = 0.0f;
    float w2 = 0.0f;
    float b  = 0.0f;

    double start_time = omp_get_wtime();

    for (int epoch = 0; epoch < epochs; epoch++) {
        float dw1 = 0.0f;
        float dw2 = 0.0f;
        float db  = 0.0f;

        // Sequential accumulation of partial gradients over N datapoints
        for (int i = 0; i < n; i++) {
            float y_pred = (w1 * x1[i]) + (w2 * x2[i]) + b;
            float err = y_pred - y[i];

            dw1 += err * x1[i];
            dw2 += err * x2[i];
            db  += err;
        }

        // Parameter update step
        w1 -= alpha * (dw1 / (float)n);
        w2 -= alpha * (dw2 / (float)n);
        b  -= alpha * (db  / (float)n);
    }

    double end_time = omp_get_wtime();

    *final_w1 = w1;
    *final_w2 = w2;
    *final_b  = b;

    return (end_time - start_time);
}

// ==========================================
// 2. OPENMP PARALLEL GRADIENT DESCENT
// ==========================================
double linear_regression_openmp(const float *x1, const float *x2, const float *y, int n, int epochs, float alpha, int num_threads, float *final_w1, float *final_w2, float *final_b) {
    float w1 = 0.0f;
    float w2 = 0.0f;
    float b  = 0.0f;

    double start_time = omp_get_wtime();

    for (int epoch = 0; epoch < epochs; epoch++) {
        float dw1 = 0.0f;
        float dw2 = 0.0f;
        float db  = 0.0f;

        // Parallel data decomposition with static schedule and reduction on gradient accumulators
        #pragma omp parallel for reduction(+:dw1, dw2, db) num_threads(num_threads) schedule(static)
        for (int i = 0; i < n; i++) {
            float y_pred = (w1 * x1[i]) + (w2 * x2[i]) + b;
            float err = y_pred - y[i];

            dw1 += err * x1[i];
            dw2 += err * x2[i];
            db  += err;
        }

        // Parameter update step
        w1 -= alpha * (dw1 / (float)n);
        w2 -= alpha * (dw2 / (float)n);
        b  -= alpha * (db  / (float)n);
    }

    double end_time = omp_get_wtime();

    *final_w1 = w1;
    *final_w2 = w2;
    *final_b  = b;

    return (end_time - start_time);
}

// ==========================================
// MAIN FUNCTION
// ==========================================
int main() {
    printf("=========================================================================\n");
    printf("     High Performance Computing - Linear Regression Optimization        \n");
    printf("=========================================================================\n");
    printf("Configuration:\n");
    printf("  Dataspace Dimension : 3D (2D Features: X1, X2 | 1D Target: Y)\n");
    printf("  Number of Points (N): %d\n", N);
    printf("  Number of Epochs    : %d\n", NUM_EPOCHS);
    printf("  Learning Rate (eta) : %.4f\n", LEARNING_RATE);
    printf("  OpenMP Threads      : %d\n", NUM_THREADS);
    printf("  True Model Equation : Y = %.2f * X1 + %.2f * X2 + %.2f\n", W1_TRUE, W2_TRUE, B_TRUE);
    printf("-------------------------------------------------------------------------\n");

    // Dynamic memory allocation on host
    float *h_x1 = (float *)malloc(N * sizeof(float));
    float *h_x2 = (float *)malloc(N * sizeof(float));
    float *h_y  = (float *)malloc(N * sizeof(float));

    if (!h_x1 || !h_x2 || !h_y) {
        printf("Error: Failed to allocate memory for dataset.\n");
        return -1;
    }

    printf("Generating synthetic 3D dataset...\n");
    generate_dataset(h_x1, h_x2, h_y, N);
    printf("Dataset generated successfully.\n\n");

    // ---------------------------------------------------------
    // 1. Sequential Execution
    // ---------------------------------------------------------
    printf("[1] Running Sequential Linear Regression...\n");
    float seq_w1, seq_w2, seq_b;
    double seq_time = linear_regression_sequential(h_x1, h_x2, h_y, N, NUM_EPOCHS, LEARNING_RATE, &seq_w1, &seq_w2, &seq_b);
    float seq_loss = compute_mse_loss(h_x1, h_x2, h_y, N, seq_w1, seq_w2, seq_b);
    printf("    Learned Model : Y = %.4f * X1 + %.4f * X2 + %.4f\n", seq_w1, seq_w2, seq_b);
    printf("    Final MSE Loss: %.6f\n", seq_loss);
    printf("    Execution Time: %.6f seconds\n\n", seq_time);

    // ---------------------------------------------------------
    // 2. OpenMP Parallel Execution
    // ---------------------------------------------------------
    printf("[2] Running OpenMP Parallel Linear Regression (%d Threads)...\n", NUM_THREADS);
    float omp_w1, omp_w2, omp_b;
    double omp_time = linear_regression_openmp(h_x1, h_x2, h_y, N, NUM_EPOCHS, LEARNING_RATE, NUM_THREADS, &omp_w1, &omp_w2, &omp_b);
    float omp_loss = compute_mse_loss(h_x1, h_x2, h_y, N, omp_w1, omp_w2, omp_b);
    printf("    Learned Model : Y = %.4f * X1 + %.4f * X2 + %.4f\n", omp_w1, omp_w2, omp_b);
    printf("    Final MSE Loss: %.6f\n", omp_loss);
    printf("    Execution Time: %.6f seconds\n", omp_time);
    printf("    Speedup vs Seq: %.2fx\n\n", seq_time / omp_time);

    // ---------------------------------------------------------
    // 3. Granularity Analysis (Thread Scaling)
    // ---------------------------------------------------------
    printf("-------------------------------------------------------------------------\n");
    printf("                    THREAD SCALABILITY COMPARISON                        \n");
    printf("-------------------------------------------------------------------------\n");
    printf("%-15s | %-12s | %-12s | %-12s\n", "Thread Count", "Time (s)", "Speedup", "Efficiency");
    printf("-------------------------------------------------------------------------\n");
    printf("%-15s | %10.6f s | %10.2fx | %10.1f%%\n", "1 (Sequential)", seq_time, 1.0, 100.0);

    int thread_counts[] = {2, 4, 8};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int k = 0; k < num_configs; k++) {
        int threads = thread_counts[k];
        float t_w1, t_w2, t_b;
        double t_time = linear_regression_openmp(h_x1, h_x2, h_y, N, NUM_EPOCHS, LEARNING_RATE, threads, &t_w1, &t_w2, &t_b);
        double speedup = seq_time / t_time;
        double efficiency = (speedup / (double)threads) * 100.0;
        
        char label[20];
        snprintf(label, sizeof(label), "%d Threads", threads);
        printf("%-15s | %10.6f s | %10.2fx | %10.1f%%\n", label, t_time, speedup, efficiency);
    }
    printf("=========================================================================\n");

    // Clean up allocated memory
    free(h_x1);
    free(h_x2);
    free(h_y);

    return 0;
}
