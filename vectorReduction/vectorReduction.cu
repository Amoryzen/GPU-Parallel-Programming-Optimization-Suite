#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cuda_runtime.h>

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
    }
}

__global__ void reduce_in_place(float *input, int n) {
    __shared__ float shared[256]; // Shared memory for each block
    
    int tid = threadIdx.x;
    int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // Load input elements into shared memory
    shared[tid] = (index < n ? input[index] : 0.0f) + (index + blockDim.x < n ? input[index + blockDim.x] : 0.0f);
    __syncthreads();
    
    // Perform in-place reduction within each block using Sequential Addressing
    // Stride starts at half the block size and decreases by half each iteration
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // Only the first 'stride' threads are active
        if (tid < stride) {
            // Check bounds to ensure we don't access memory outside 'n'
            if (index + stride < n) {
                shared[tid] += shared[tid + stride]; 
            }
        }
        __syncthreads();
    }

    // Write the result of this block to the first element of the block
    if (tid == 0) {
        input[blockIdx.x] = shared[0];
    }
}

// Function to calculate the sum of an array on the host
float cpu_reduce(float *input, int n) {
    float sum = 0.0;
    
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }

    return sum;
}

int main() {
    int n = 1024 * 1024;
    size_t bytes = n * sizeof(float);

    // Allocate space for the input vector
    float *h_input = new float[n];
    float *d_input;

    // Initialize input array
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i+1);
    }

    cudaCheckError(cudaMalloc(&d_input, bytes)); // Allocate device memory  

    // Copy input vector to device
    cudaCheckError(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));    

    // Launch the reduction kernel multiple times
    int block_size = 256; // Number of threads per block
    int grid_size = (n + 2 * block_size - 1) / (2 * block_size); // Number of blocks
    size_t shared_mem_size = block_size * sizeof(float); // Shared memory size per block

    float total_sum = cpu_reduce(h_input, n); // Calculate the sum on the host
    printf("Total sum (CPU): %f\n", total_sum);

    // Perform iterative reduction until only one block remains
    while (grid_size > 1) {
        // Launch kernel
        reduce_in_place <<< grid_size, block_size, shared_mem_size >>> (d_input, n);
        cudaCheckError(cudaDeviceSynchronize()); // Synchronize with the CPU

        // Update n and grid size for the next iteration
        n = grid_size;
        grid_size = (n + 2 * block_size - 1) / (2 * block_size);
    }
    
    // Final reduction when grid_size is 1
    reduce_in_place <<< 1, block_size, shared_mem_size >>> (d_input, n);
    cudaCheckError(cudaDeviceSynchronize()); // Synchronize with the CPU

    gpuKernelCheck();    

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_input, d_input,  sizeof(float), cudaMemcpyDeviceToHost));

    // Print result
    printf("Final sum (GPU): %f\n", h_input[0]);   

    // Free memory
    cudaCheckError(cudaFree(d_input));
    delete[] h_input;

    return 0;
}
