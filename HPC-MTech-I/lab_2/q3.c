/*
Design and develop a parallel solution of matrix matrix multiplication problem. 
Compare and contrast the performance of coarse-grained and 
fined-grained data decomposition methods. 
For your input, use two sufficiently large sized input matrices.
*/


#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 1500
#define NUM_THREADS 8

int a[N][N];
int b[N][N];
int c[N][N];


void generate_matrices(bool is_print_matrix){
    int i,j;
    srand(time(NULL));
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            a[i][j] = rand()%10;
            b[i][j] = rand()%10;
        }
    }
    if(!is_print_matrix)
        return;

    printf("Matrix A\n");
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
    
    printf("Matrix B\n");
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            printf("%d ", b[i][j]);
        }
        printf("\n");
    }
}


void print_matrix(char *type){
    int i,j;
    printf("Matrix C generated after %s matrix multiplication :\n", type);
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }
}


void fine_grained_matrix_multiplication(char *type){
    int i,j,k;
    double start_time = omp_get_wtime();

    #pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            c[i][j] = 0;
            for(k=0; k<N; k++){
                c[i][j] += a[i][k]*b[k][j];
            }
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;
    printf("Time taken by %s matrix multiplication: %.6f seconds\n", type, time_taken);
}


void coarse_grained_matrix_multiplication(char *type){
    int i,j,k;
    double start_time = omp_get_wtime();

    #pragma omp parallel for private(j, k) num_threads(NUM_THREADS)
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            c[i][j] = 0;
            for(k=0; k<N; k++){
                c[i][j] += a[i][k]*b[k][j];
            }
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;
    printf("Time taken by %s matrix multiplication: %.6f seconds\n", type, time_taken);
}


int main(){
    char *type;
    bool is_print_matrix = false;
    generate_matrices(is_print_matrix);
    type="coarse grained";
    coarse_grained_matrix_multiplication(type);

    if(is_print_matrix)
        print_matrix(type);
    
    type="fine grained";
    fine_grained_matrix_multiplication(type);
    
    if(is_print_matrix)
        print_matrix(type);
    
    return 0;
}