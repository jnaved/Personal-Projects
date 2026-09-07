#include <stdio.h>
#include <stdlib.h>

int main() {
    int n;
    printf("Enter the size of the vectors: ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Invalid vector size.\n");
        return 1;
    }

    int *v1 = (int *)malloc(n * sizeof(int));
    int *v2 = (int *)malloc(n * sizeof(int));

    if (v1 == NULL || v2 == NULL) {
        printf("Memory allocation failed.\n");
        free(v1);
        free(v2);
        return 1;
    }

    printf("Enter %d elements of first vector:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &v1[i]);
    }

    printf("Enter %d elements of second vector:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &v2[i]);
    }

    long long dot_product = 0;
    for (int i = 0; i < n; i++) {
        dot_product += (long long)v1[i] * v2[i];
    }

    printf("Dot product of the two vectors: %lld\n", dot_product);

    free(v1);
    free(v2);

    return 0;
}
