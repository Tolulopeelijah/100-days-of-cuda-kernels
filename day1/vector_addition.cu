#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to perform vector addition
__global__ void vecAdd(float *d_A, float *d_B, float *d_C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Compute global index
    if (i < n) {
        d_C[i] = d_A[i] + d_B[i];
    
}

// Host function
int main() {
    int n = 1000000;  // Array size
    size_t size = n * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize input vectors with some values
    for (int i = 0; i < n; i++) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 2.0f// %%writefile vector_addition.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX; 
        h_B[i] = (float)rand() / RAND_MAX;
    }
    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch kernel
    vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print some results
    printf("C:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);

    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
;
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define CUDA kernel launch parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize; // Ensure we cover all elements

    // Launch CUDA kernel
    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify correctness


    int correct = 1;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != (h_A[i] + h_B[i])) {
            correct = 0;
            break;
        }
    }
    
    if (correct){
        printf("Vector addition completed successfully!\n");
	}
    else{
        printf("Error: Incorrect results detected.\n");
	}

    printf("Testing:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);

    }
    
    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX; 
        h_B[i] = (float)rand() / RAND_MAX;
    }
    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch kernel
    vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print some results
    printf("C:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);

    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}