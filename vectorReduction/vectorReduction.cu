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

// Function to perform warp-level reduction
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_in_place(float *input, int n) {
    __shared__ float shared[256]; // Shared memory for each block
    
    int tid = threadIdx.x;
    int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // Load input elements into shared memory
    shared[tid] = (index < n ? input[index] : 0.0f) + (index + blockDim.x < n ? input[index + blockDim.x] : 0.0f);
    __syncthreads();
    
    // Perform in-place reduction within each block using Sequential Addressing
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride]; 
        }

        __syncthreads();
    }

    // Unroll the last warp to avoid synchronization overhead
    if (tid < 32) {
        warpReduce(shared, tid);
    }

    // Write the result of this block to global memory
    if (tid == 0) {
        input[blockIdx.x] = shared[0];
    }
}

// Function to calculate the sum of an array on the host
float cpu_reduce(float *input, int size) {
    float sum = 0.0;
    
    for (int i = 0; i < size; i++) {
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

    // Allocate device memory
    cudaCheckError(cudaMalloc(&d_input, bytes)); 

    // Copy input vector to device
    cudaCheckError(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));    

    // Launch parameters
    int block_size = 256; 
    int grid_size = (n + 2 * block_size - 1) / (2 * block_size); 
    size_t shared_mem_size = block_size * sizeof(float); // This isn't actually used in kernel call since we declared __shared__ inside

    float total_sum = cpu_reduce(h_input, n); 
    printf("Total sum (CPU): %f\n", total_sum);

    // Perform iterative reduction
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
