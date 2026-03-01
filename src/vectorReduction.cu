#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define cudaCheckError(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);

    if (abort)
      exit(code);
  }
}

// Macro for checking kernel errors
#define gpuKernelCheck()                                                       \
  {                                                                            \
    gpuKernelAssert(__FILE__, __LINE__);                                       \
  }
inline void gpuKernelAssert(const char *file, int line, bool abort = true) {
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s %s %d\n", cudaGetErrorString(err),
            file, line);

    if (abort)
      exit(err);
  }
}

// Function to perform warp-level reduction
__inline__ __device__ void warpReduce(float val) {
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
}

__global__ void reduce_with_warp_optimization(float *input, int n) {
  extern __shared__ float shared[]; // Shared memory for each block
  int tid = threadIdx.x;
  int index = 2 * blockIdx.x * blockDim.x + tid;
  float sum = 0.0f;

  sum = (index < n ? input[index] : 0.0f) +
        (index + blockDim.x < n
             ? input[index + blockDim.x]
             : 0.0f); // Load two elements from global memory into register

  // Perform in-place reduction within each block using Sequential Addressing
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  // Write the result of each warp to shared memory
  if (tid % warpSize == 0) {
    shared[tid / warpSize] = sum;
  }

  __syncthreads();

  // Perform warp-level reduction on the shared memory
  if (tid < warpSize) {
    sum = (tid < (blockDim.x / warpSize)) ? shared[tid] : 0.0f;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }

  // Write the result of this block to global memory
  if (tid == 0) {
    input[blockIdx.x] = sum;
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

int main(int argc, char **argv) {
  int n = 1024 * 1024;
  int block_size = 256;

  printf("Enter vector size (n): ");
  scanf("%d", &n);

  printf("Enter block size: ");
  scanf("%d", &block_size);

  printf("Running with n=%d, block_size=%d\n", n, block_size);

  size_t bytes = n * sizeof(float);

  // Allocate space for the input vector
  float *h_input = new float[n];
  float *d_input;

  // Initialize input array
  for (int i = 0; i < n; i++) {
    h_input[i] = static_cast<float>(i + 1);
  }

  // Allocate device memory
  cudaCheckError(cudaMalloc(&d_input, bytes));

  // Copy input vector to device
  cudaCheckError(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  // Launch parameters
  // int block_size = 256; // Defined at start of main
  int grid_size = (n + 2 * block_size - 1) / (2 * block_size);
  size_t shared_mem_size =
      (block_size / 32) *
      sizeof(float); // This isn't actually used in kernel call since we
                     // declared __shared__ inside

  float total_sum = cpu_reduce(h_input, n);
  printf("Total sum (CPU): %f\n", total_sum);

  // Perform iterative reduction
  while (grid_size > 1) {
    // Launch kernel
    reduce_with_warp_optimization<<<grid_size, block_size, shared_mem_size>>>(
        d_input, n);
    cudaCheckError(cudaDeviceSynchronize()); // Synchronize with the CPU

    // Update n and grid size for the next iteration
    n = grid_size;
    grid_size = (n + 2 * block_size - 1) / (2 * block_size);
  }

  // Final reduction when grid_size is 1
  reduce_with_warp_optimization<<<1, block_size, shared_mem_size>>>(d_input, n);
  cudaCheckError(cudaDeviceSynchronize()); // Synchronize with the CPU

  gpuKernelCheck();

  // Copy result back to host
  cudaCheckError(
      cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost));

  // Print result
  printf("Final sum (GPU): %f\n", h_input[0]);

  // Free memory
  cudaCheckError(cudaFree(d_input));
  delete[] h_input;

  return 0;
}
