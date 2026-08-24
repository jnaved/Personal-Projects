/*
Implement an omp critical section based solution for race condition 
occurred in the previous question.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_THREADS 4
#define ITERATIONS 50000

void demonstrate_critical_block(){
    long long shared_counter = 0;
    long long expected_value = (long long)NUM_THREADS*ITERATIONS;
    
    printf("Number of Threads : %d\n", NUM_THREADS);
    printf("Number of Iteration : %d\n", ITERATIONS);
    printf("Expected Value : %lld\n", expected_value);
    
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        for(int i = 0; i < ITERATIONS; i++){
            #pragma omp critical
            {
                shared_counter+=1;
            }
        }
        printf("Thread ID : %d, thread's final shared counter value : %lld\n", tid, shared_counter);
    }

    printf("Actual Value : %lld\n", shared_counter);
}

int main(){
    demonstrate_critical_block();
    return 0;
}