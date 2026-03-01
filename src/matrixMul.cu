#include <cmath>
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

// CPU Kernel for matrix multiplication
void matrixMulCPU(float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0;

      for (int k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * n + j];
      }

      C[i * n + j] = sum;
    }
  }
}

// GPU Kernel for matrix multiplication (tiled + float4 loads)
// blockDim: (4,16) => 4 threads in x-axis, each loading 4 columns => 16 columns
// per block
//                    16 threads in y-axis, each loading 16 rows => 256 rows per
//                    block
__global__ void matrixMulGPU(float *__restrict__ A, float *__restrict__ B,
                             float *__restrict__ C, int n, int tile_size) {
  // Declare shared memory for tiles
  extern __shared__ float shared_mem[];
  float *tileA = shared_mem;
  float *tileB = shared_mem + tile_size * tile_size;

  // Identify which row & column(s) this thread is responsible for
  int row = blockIdx.y * tile_size + threadIdx.y; // each thread handles 1 row
  int base_col = blockIdx.x * tile_size +
                 (threadIdx.x * 4); // each thread handles 4 columns

  float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // Initialize sum for this thread

  int num_tiles = (n + tile_size - 1) / tile_size; // Number of tiles to process

  for (int t = 0; t < num_tiles; t++) {
    // Load sub-tile of A and B from global memory to shared memory

    // A
    {
      // Global column for A
      int global_A_col = t * tile_size + (threadIdx.x * 4);
      const float *A_ptr = A + row * n + global_A_col;
      float *A_dst = &tileA[threadIdx.y * tile_size + threadIdx.x * 4];

      // Load 4 elements from global memory to shared memory
      // Note: This assumes 16-byte alignment which requires N and tile_size to
      // be multiples of 4
      float4 A_vec = *reinterpret_cast<const float4 *>(A_ptr);
      *reinterpret_cast<float4 *>(A_dst) = A_vec;
    }

    // B
    {
      // Global row for B
      int global_B_row = t * tile_size + threadIdx.y;
      const float *B_ptr = B + global_B_row * n + base_col;
      float *B_dst = &tileB[threadIdx.y * tile_size + threadIdx.x * 4];

      // Load 4 elements from global memory to shared memory
      float4 B_vec = *reinterpret_cast<const float4 *>(B_ptr);
      *reinterpret_cast<float4 *>(B_dst) = B_vec;
    }

    __syncthreads(); // Synchronize threads to ensure all tiles are loaded

    // Compute partial sum
    for (int k = 0; k < tile_size; k++) {
      float A_val = tileA[threadIdx.y * tile_size + k]; // 1 scalar
      float4 B_vec = *reinterpret_cast<const float4 *>(
          &tileB[k * tile_size + threadIdx.x * 4]); // 4 scalars

      // 1 scalar * 4 scalars = 4 scalars
      sum[0] += A_val * B_vec.x;
      sum[1] += A_val * B_vec.y;
      sum[2] += A_val * B_vec.z;
      sum[3] += A_val * B_vec.w;
    }

    __syncthreads(); // Synchronize threads to ensure all tiles are processed
  }

  // Write the final partial sums
  if (row < n && base_col < n) {
    float *C_out = C + (row * n + base_col);

    C_out[0] = sum[0];
    C_out[1] = sum[1];
    C_out[2] = sum[2];
    C_out[3] = sum[3];
  }
}

int main() {
  int n, tile_size;

  // Receive inputs from the user
  printf("Enter N! "); // N = 2048 for 100% occupancy
  scanf("%d", &n);
  printf("Enter tile size! "); // tile_size = 32 for 100% occupancy
  scanf("%d", &tile_size);

  // Validate inputs
  if (n % 4 != 0 || tile_size % 4 != 0) {
    printf("Error: N and tile size must be multiples of 4.\n");
    return -1;
  }

  // Allocate host memory
  float *h_A = new float[n * n];
  float *h_B = new float[n * n];
  float *h_C = new float[n * n];
  float *h_C_gpu = new float[n * n];

  // Initialize input matrices
  srand(0);
  for (int i = 0; i < n * n; i++) {
    h_A[i] = float(rand() % 10);
    h_B[i] = float(rand() % 10);
    h_C[i] = 0.0f;
    h_C_gpu[i] = 0.0f;
  }

  matrixMulCPU(h_A, h_B, h_C, n); // CPU baseline

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  size_t size = sizeof(float) * n * n;

  cudaCheckError(cudaMalloc(&d_A, size));
  cudaCheckError(cudaMalloc(&d_B, size));
  cudaCheckError(cudaMalloc(&d_C, size));

  // Copy data from host to device
  cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // Configure kernel launch
  dim3 block_size(tile_size / 4, tile_size);
  dim3 grid_size((n + tile_size - 1) / tile_size,
                 (n + tile_size - 1) / tile_size);
  size_t shmem_size = 2 * tile_size * tile_size * sizeof(float);

  matrixMulGPU<<<grid_size, block_size, shmem_size>>>(
      d_A, d_B, d_C, n, tile_size); // Launch kernel

  cudaCheckError(cudaDeviceSynchronize());

  cudaCheckError(
      cudaMemcpy(h_C_gpu, d_C, size,
                 cudaMemcpyDeviceToHost)); // Copy result from device to host

  // Compare results
  long long diff = 0;

  for (int i = 0; i < n; i++) {
    diff += (long long)fabs(h_C_gpu[i] - h_C[i]);
  }

  if (diff == 0)
    printf("Correct!\n");
  else
    printf("Incorrect! diff = %lld\n", diff);

  // Free memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_gpu;

  cudaCheckError(cudaFree(d_A));
  cudaCheckError(cudaFree(d_B));
  cudaCheckError(cudaFree(d_C));

  return 0;
}
