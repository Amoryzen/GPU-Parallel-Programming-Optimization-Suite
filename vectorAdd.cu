#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_SIZE 1024*1024*1024 // Total elements in the vector
#define CHUNK_SIZE 1024*1024*128 // Chunk size for each kernel launch
#define BLOCK_SIZE 1024 // Number of threads per block

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int chunkSize) {
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
    // Allocate memory space
    int *chunk_a, *chunk_b, *chunk_c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors 
    size_t chunkSizeBytes = CHUNK_SIZE * sizeof(int); // Size of each vector in bytes

    // Allocate and initialize host vectors
    chunk_a = (int*)malloc(chunkSizeBytes);
    chunk_b = (int*)malloc(chunkSizeBytes);
    chunk_c = (int*)malloc(chunkSizeBytes);

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, chunkSizeBytes);
    cudaMalloc((void**)&d_b, chunkSizeBytes);
    cudaMalloc((void**)&d_c, chunkSizeBytes);

    // Calculate the number of blocks for the kernel
    int numBlocks = (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Process the vector in chunks
    for (long long offset = 0; offset < TOTAL_SIZE; offset += CHUNK_SIZE) {
        // Calculate the current chunk size
        int currentChunkSize = CHUNK_SIZE;

        if (offset + CHUNK_SIZE > TOTAL_SIZE) {
            currentChunkSize = TOTAL_SIZE - offset;
        }

        printf("Offset: %lld, Current Chunk Size: %d\n", offset, currentChunkSize);

        // Populate the chunks with random data
        random_inst(chunk_a, currentChunkSize);
        random_inst(chunk_b, currentChunkSize);

        // Copy chunks to the GPU
        cudaMemcpy(d_a, chunk_a, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, chunk_b, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice);
        
        cudaEventRecord(start);
        
        vectorAdd <<< numBlocks, BLOCK_SIZE >>> (d_a, d_b, d_c, currentChunkSize); // Launch the kernel

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time elapsed: %f ms\n", milliseconds);

        // Copy the result back to the host
        cudaMemcpy(chunk_c, d_c, currentChunkSize * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nPrint elements of the result:\n");
    for (int i = 0; i < 100; i++) {
        printf("Element %d: %d + %d = %d\n", i, chunk_a[i], chunk_b[i], chunk_c[i]);
    }

    // Step 7: Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(chunk_a);
    free(chunk_b);
    free(chunk_c);  

    cudaDeviceSynchronize();

    return 0;
}
