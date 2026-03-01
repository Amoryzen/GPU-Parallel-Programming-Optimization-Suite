#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

using namespace nvcuda::wmma;

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

#define TILE_SIZE 16

// GPU kernel: half-precision inputs, float accumulation/outputs
// 1D block of 128 threads => 4 warps per block
// Each warp computes a 16x16 tile in the output
__global__ void matrixMulGPU(const half *A, const half *B, float *C, int n) {
  int warps_per_block =
      blockDim.x / 32; // warps_per_block = 4 if blockDim.x = 128
  int warp_id =
      (blockIdx.x * warps_per_block) + (threadIdx.x / 32); // Global warp ID
  int tiles_per_dim = n / TILE_SIZE; // Number of tiles per dimension
  int tile_count = tiles_per_dim * tiles_per_dim; // Total number of tiles

  if (warp_id >= tile_count)
    return; // Exit if warp_id is out of bounds

  // Identify which 16x16 tile this warp is responsible for
  int tile_row = warp_id / tiles_per_dim; // Row index of the tile
  int tile_col = warp_id % tiles_per_dim; // Column index of the tile

  // Create WMMA fragments
  fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, col_major> frag_a;
  fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, row_major> frag_b;
  fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> frag_c;

  fill_fragment(frag_c, 0.0f); // Initialize accumulator fragment to zero

  // Loop over all tiles in the matrix in steps of 16
  for (int tile_k = 0; tile_k < n; tile_k += 16) {
    // Calculate the global memory addresses for the current tile
    const half *A_tile = A + tile_row * TILE_SIZE * n + tile_k;
    const half *B_tile = B + tile_k * n + tile_col * TILE_SIZE;

    // Load matrix tiles from global memory to fragments
    load_matrix_sync(frag_a, A_tile, n);
    load_matrix_sync(frag_b, B_tile, n);

    // Perform matrix multiplication
    mma_sync(frag_c, frag_a, frag_b, frag_c);
  }

  // Write the 16x16 tile to global memory
  float *C_tile = C + tile_row * TILE_SIZE * n + tile_col * TILE_SIZE;
  store_matrix_sync(C_tile, frag_c, n, mem_row_major);
}

// Simple helper for filling a half-precision matrix with random values [0..1]
void fillWithRandom(half *matrix, int size) {
  for (int i = 0; i < size; i++) {
    float r = static_cast<float>(rand()) / RAND_MAX;
    matrix[i] = __float2half(r);
  }
}

// CPU Kernel for matrix multiplication, but store result in float
void matrixMulCPU(const half *A, const half *B, float *C, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;

      for (int k = 0; k < n; k++) {
        float val_a = __half2float(A[i * n + k]);
        float val_b = __half2float(B[k * n + j]);
        sum += val_a * val_b;
      }

      C[i * n + j] = sum;
    }
  }
}

int main() {
  int N;

  printf("Enter N: "); // N = 2048 for 100% occupancy
  if (scanf("%d", &N) != 1) {
    printf("Error reading N.\n");
    return 1;
  }

  // N must be a multiple of 16
  if (N % TILE_SIZE != 0) {
    printf("N must be a multiple of %d!\n", TILE_SIZE);
    return 1;
  }

  srand(0);

  // Allocate host memory: half inputs, float output
  size_t total_elements = N * N;
  size_t bytes_a_or_b = total_elements * sizeof(half);
  size_t bytes_c = total_elements * sizeof(float);

  half *h_A = (half *)malloc(bytes_a_or_b);
  half *h_B = (half *)malloc(bytes_a_or_b);

  float *h_C_gpu = (float *)malloc(bytes_c); // GPU result
  float *h_C_cpu = (float *)malloc(bytes_c); // CPU result

  // Initialize input matrices
  fillWithRandom(h_A, N * N);
  fillWithRandom(h_B, N * N);

  // Allocate device memory
  half *d_A = nullptr;
  half *d_B = nullptr;
  float *d_C = nullptr;

  cudaCheckError(cudaMalloc(&d_A, bytes_a_or_b));
  cudaCheckError(cudaMalloc(&d_B, bytes_a_or_b));
  cudaCheckError(cudaMalloc(&d_C, bytes_c));

  // Copy data from host to device
  cudaCheckError(cudaMemcpy(d_A, h_A, bytes_a_or_b, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, h_B, bytes_a_or_b, cudaMemcpyHostToDevice));
  cudaCheckError(
      cudaMemset(d_C, 0, bytes_c)); // Initialize device memory to zero

  // Compute grid/block configuration
  //  - 128 threads => 4 warps per block
  //  - Total 16x16 tiles = (N/16) * (N/16)
  //  - Each block handles 4 tiles
  int tiles_per_dim = N / TILE_SIZE;
  int tile_count = tiles_per_dim * tiles_per_dim;
  int warps_per_block = 128 / 32;
  int blocks_needed = (tile_count + warps_per_block - 1) / warps_per_block;

  dim3 block_size(128);
  dim3 grid_size(blocks_needed);

  matrixMulGPU<<<grid_size, block_size>>>(d_A, d_B, d_C,
                                          N); // Launch the GPU kernel

  cudaCheckError(cudaDeviceSynchronize());

  cudaCheckError(
      cudaMemcpy(h_C_gpu, d_C, bytes_c,
                 cudaMemcpyDeviceToHost)); // Copy result from device to host

  matrixMulCPU(h_A, h_B, h_C_cpu, N); // Run CPU kernel for comparison

  // Compare the first 10 elements of the result
  printf("Matrix multiplication (FP16 x FP16 -> FP32) completed!\n");
  printf("Comparing the first 10 elements of the result (CPU vs GPU):\n");

  for (int i = 0; i < 10; i++) {
    float val_cpu = h_C_cpu[i];
    float val_gpu = h_C_gpu[i];

    printf("h_C_gpu[%d] = %f\n", i, val_gpu);
    printf("h_C_cpu[%d] = %f\n", i, val_cpu);
  }

  // Free device memory
  cudaCheckError(cudaFree(d_A));
  cudaCheckError(cudaFree(d_B));
  cudaCheckError(cudaFree(d_C));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C_gpu);
  free(h_C_cpu);

  return 0;
}
