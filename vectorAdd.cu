#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

// Declare the kernel
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Ensure we don't go out of bounds
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void random_inst(int *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = rand() % 100;
    }
}

// Declare the main function
int main() {
    // Define vector size and block size
    int n;
    int block_size;

    // Receive inputs from the user
    printf("Enter vector size: ");
    scanf("%d", &n);
    printf("Enter block size: ");
    scanf("%d", &block_size);
    
    size_t bytes = n * sizeof(int); // Size of each vector in bytes
    
    // Initialize and allocate host vectors
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    
    // Populate the chunks with random data
    random_inst(h_a, n);
    random_inst(h_b, n);

    // Initialize and allocate device vectors
    int *d_a, *d_b, *d_c; // Device vectors 
    cudaCheckError(cudaMalloc((void**)&d_a, bytes));
    cudaCheckError(cudaMalloc((void**)&d_b, bytes));
    cudaCheckError(cudaMalloc((void**)&d_c, bytes));

    // Copy vectors to the GPU
    cudaCheckError(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));
    
    int numBlocks = (n + block_size - 1) / block_size; // Calculate the number of blocks for the kernel
    
    cudaFuncSetCacheConfig(vectorAdd, cudaFuncCachePreferL1); // 

    cudaCheckError(cudaEventRecord(start)); // Start recording
    
    vectorAdd <<< numBlocks, block_size >>> (d_a, d_b, d_c, n); // Launch the kernel
    
    gpuKernelCheck();
    
    cudaCheckError(cudaEventRecord(stop)); // Stop recording
    cudaCheckError(cudaEventSynchronize(stop)); // Synchronize event

    // Calculate and print the execution time
    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time elapsed: %f ms\n", milliseconds);

    cudaCheckError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost)); // Copy the result back to the host

    // Destroy events
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));

    // Print the elements and their results
    printf("\nElements & Results:\n");
    for (int i = 0; i < 100; i++) {
        printf("Element %d: %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
    }

    // Free device memory
    cudaCheckError(cudaFree(d_a));
    cudaCheckError(cudaFree(d_b));
    cudaCheckError(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);  

    cudaCheckError(cudaDeviceSynchronize()); // Synchronize with the CPU

    return 0;
}
