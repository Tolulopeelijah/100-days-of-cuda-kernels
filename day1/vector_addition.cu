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
        h_B[i] = i * 2.0f;
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

    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
