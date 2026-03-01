#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

// Declare the kernel
__global__ void matrixAdd(float *a, float *b, float *c, int m, int n) {
  // Calculate the column and row indices of the matrix element
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Make sure we do not go out of bounds
  if (col < m && row < n) {
    // Calculate the 1D index for the 2D element
    int idx = col + row * m;

    // Perform element-wise addition
    c[idx] = a[idx] + b[idx];
  }
}

// Declare the main function
int main() {
  // Declare essential parameters
  int m, n;
  int block_x, block_y;

  // Receive inputs from the user
  printf("Enter matrix width (M/column): "); // M = 1024 for 100% occupancy
  scanf("%d", &m);
  printf("Enter matrix height (N/column): "); // N = 1024 for 100% occupancy
  scanf("%d", &n);
  printf("Enter block size X: "); // X = 16 for 100% occupancy
  scanf("%d", &block_x);
  printf("Enter block size Y: "); // Y = 16 for 100% occupancy
  scanf("%d", &block_y);

  size_t size = (size_t)m * n * sizeof(float);

  // Initialize and allocate host matrices
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  // Initialize matrices a and b randomly
  for (int i = 0; i < m * n; i++) {
    // Randomize values between 0 and 1
    h_a[i] = 10 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    h_b[i] = 10 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  // Initialize and allocate device matrices
  float *d_a, *d_b, *d_c;
  cudaCheckError(cudaMalloc((void **)&d_a, size));
  cudaCheckError(cudaMalloc((void **)&d_b, size));
  cudaCheckError(cudaMalloc((void **)&d_c, size));

  // Copy matrices a and b to the device
  cudaCheckError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  // Define block and grid dimensions
  dim3 blockDim(block_x, block_y);
  dim3 gridDim((m + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);

  cudaFuncSetCacheConfig(matrixAdd, cudaFuncCachePreferL1); //

  cudaCheckError(cudaEventRecord(start));

  matrixAdd<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n); // Launch the kernel

  gpuKernelCheck();

  cudaCheckError(cudaEventRecord(stop));      // Stop recording
  cudaCheckError(cudaEventSynchronize(stop)); // Synchronize event

  // Calculate and print the execution time
  float milliseconds = 0;
  cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Time elapsed: %f ms\n", milliseconds);

  cudaCheckError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  // Destroy events
  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));

  // Print the elements and their results
  printf("a[i] + b[i] = c[i]\n");
  for (int i = 0; i < 50; i++) {
    printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
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
