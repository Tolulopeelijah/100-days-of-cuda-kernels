#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // CUDA Block size

// CUDA Kernel for Matrix Addition
__global__ void matrixAdd(float *C, const float *A, const float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Compute row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Compute column index

    if (row < N && col < N) {  // Ensure within matrix bounds
        int index = row * N + col;  // Convert (row, col) to 1D index
        C[index] = A[index] + B[index];  // Perform element-wise addition
    }
}

// Host Function to Launch CUDA Kernel
void launchMatrixAdd(float *h_C, const float *h_A, const float *h_B, int N) {
    int size = N * N * sizeof(float);  // Compute memory size

    // Device memory pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);  // Threads per block
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // Grid size

    // Launch CUDA kernel
    matrixAdd<<<gridDim, blockDim>>>(d_C, d_A, d_B, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 4;  // Matrix dimension (N x N)
    float h_A[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C[16];  // Output matrix

    launchMatrixAdd(h_C, h_A, h_B, N);  // Call matrix addition function

    // Print the result
    printf("Resultant Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%0.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}
