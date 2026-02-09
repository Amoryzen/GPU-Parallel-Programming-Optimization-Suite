#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for checking CUDA errors
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort) exit(code);
    }
}

// Macro for checking kernel errors
#define gpuKernelCheck() { gpuKernelAssert(__FILE__, __LINE__); }

inline void gpuKernelAssert(const char *file, int line, bool abort = true) {
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s %s %d\n", cudaGetErrorString(err), file, line);

        if (abort) exit(err);
    }}

// Kernel
__global__ void vectorAdd(int *a, int *b, int *c, int chunkSize) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Ensure we don't go out of bounds
    if (i < chunkSize) {
        c[i] = a[i] + b[i];
    }
}

void random_inst(int *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = rand() % 100;
    }
}

int main() {
    // Define vector size, chunk size, and block size
    long long total_size;
    int chunk_size;
    int block_size;

    printf("Enter total size: ");
    scanf("%lld", &total_size);

    printf("Enter chunk size: ");
    scanf("%d", &chunk_size);

    printf("Enter block size: ");
    scanf("%d", &block_size);

    // Allocate memory space
    int *chunk_a, *chunk_b, *chunk_c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors 
    size_t chunkSizeBytes = chunk_size * sizeof(int); // Size of each vector in bytes

    // Allocate and initialize host vectors
    chunk_a = (int*)malloc(chunkSizeBytes);
    chunk_b = (int*)malloc(chunkSizeBytes);
    chunk_c = (int*)malloc(chunkSizeBytes);

    // Allocate memory on the GPU
    cudaCheckError(cudaMalloc((void**)&d_a, chunkSizeBytes));
    cudaCheckError(cudaMalloc((void**)&d_b, chunkSizeBytes));
    cudaCheckError(cudaMalloc((void**)&d_c, chunkSizeBytes));

    int numBlocks = (chunk_size + block_size - 1) / block_size; // Calculate the number of blocks for the kernel

    // Create events for timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    // Process the vector in chunks
    for (long long offset = 0; offset < total_size; offset += chunk_size) {
        int currentChunkSize = chunk_size; // Calculate the current chunk size

        if (offset + chunk_size > total_size) {
            currentChunkSize = total_size - offset;
        }

        printf("Offset: %lld, current chunk size: %d, number of blocks: %d\n", offset, currentChunkSize, numBlocks);

        // Populate the chunks with random data
        random_inst(chunk_a, currentChunkSize);
        random_inst(chunk_b, currentChunkSize);

        // Copy chunks to the GPU
        cudaCheckError(cudaMemcpy(d_a, chunk_a, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_b, chunk_b, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice));
        
        cudaCheckError(cudaEventRecord(start)); // Start recording
        
        vectorAdd <<< numBlocks, block_size >>> (d_a, d_b, d_c, currentChunkSize); // Launch the kernel
        
        gpuKernelCheck();
        
        cudaCheckError(cudaEventRecord(stop)); // Stop recording

        cudaCheckError(cudaEventSynchronize(stop)); // Synchronize event

        float milliseconds = 0;
        
        // Calculate and print the execution time
        cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("Time elapsed: %f ms\n", milliseconds);

        cudaCheckError(cudaMemcpy(chunk_c, d_c, currentChunkSize * sizeof(int), cudaMemcpyDeviceToHost)); // Copy the result back to the host
    }

    // Destroy events
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));

    // Print the elements and their results
    printf("\nElements & Results:\n");
    for (int i = 0; i < 100; i++) {
        printf("Element %d: %d + %d = %d\n", i, chunk_a[i], chunk_b[i], chunk_c[i]);
    }

    // Cleanup
    cudaCheckError(cudaFree(d_a));
    cudaCheckError(cudaFree(d_b));
    cudaCheckError(cudaFree(d_c));

    free(chunk_a);
    free(chunk_b);
    free(chunk_c);  

    cudaCheckError(cudaDeviceSynchronize()); // Synchronize with the CPU

    return 0;
}
