#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>


static const int N = 512;        // size of matrix
static const int TILE_SIZE = 16; // size of tile

// CPU Kernel for matrix multiplication
void matrixMulCPU(float *A, float *B, float *C) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;

      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }

      C[i * N + j] = sum;
    }
  }
}

// GPU Kernel for matrix multiplication (tiled + float4 loads)
// blockDim: (4,16) => 4 threads in x-axis, each loading 4 columns => 16 columns
// per block
//                    16 threads in y-axis, each loading 16 rows => 256 rows per
//                    block
__global__ void matrixMulGPU(float *__restrict__ A, float *__restrict__ B,
                             float *__restrict__ C, int N) {
  // Declare shared memory for tiles
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  // Identify which row & column(s) this thread is responsible for
  // blockDim.x, blockDim.y: number of threads in x-axis & y-axis

  // blockIdx.x, blockIdx.y: which tile in the grid
  // blockIdx.x in [0..127], blockIdx.y in [0..31]

  // threadIdx.x, threadIdx.y: which thread in the block
  // threadIdx.x in [0..3], threadIdx.y in [0..15]
  int row = blockIdx.y * TILE_SIZE + threadIdx.y; // each thread handles 1 row
  int base_col = blockIdx.x * TILE_SIZE +
                 (threadIdx.x * 4); // each thread handles 4 columns

  float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // Initialize sum for this thread

  int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE; // Number of tiles to process

  for (int t = 0; t < num_tiles; t++) {
    // Load sub-tile of A and B from global memory to shared memory
    // Each thread loads a float4 from A, and a float4 from B

    // A
    // tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE +
    // threadIdx.x)];
    {
      // Global column for A => t * TILE_SIZE + threadIdx.x
      int global_A_col = t * TILE_SIZE + (threadIdx.x * 4);
      const float *A_ptr = A + row * N + global_A_col;
      float *A_dst = &tileA[threadIdx.y][threadIdx.x * 4];

      // Load 4 elements from global memory to shared memory
      float4 A_vec = *reinterpret_cast<const float4 *>(A_ptr);
      *reinterpret_cast<float4 *>(A_dst) = A_vec;
    }

    // B
    // tileB[threadIdx.y][threadIdx.x] = B[row * N + (t * TILE_SIZE +
    // threadIdx.x)];
    {
      // Global row for B => t * TILE_SIZE + threadIdx.y
      int global_B_row = t * TILE_SIZE + threadIdx.y;
      const float *B_ptr = B + global_B_row * N + base_col;
      float *B_dst = &tileB[threadIdx.y][threadIdx.x * 4];

      // Load 4 elements from global memory to shared memory
      float4 B_vec = *reinterpret_cast<const float4 *>(B_ptr);
      *reinterpret_cast<float4 *>(B_dst) = B_vec;
    }

    __syncthreads(); // Synchronize threads to ensure all tiles are loaded

    // Compute partial sum
    for (int k = 0; k < TILE_SIZE; k++) {
      float A_val = tileA[threadIdx.y][k]; // 1 scalar
      float4 B_vec = *reinterpret_cast<const float4 *>(
          &tileB[k][threadIdx.x * 4]); // 4 scalars

      // 1 scalar * 4 scalars = 4 scalars
      sum[0] += A_val * B_vec.x;
      sum[1] += A_val * B_vec.y;
      sum[2] += A_val * B_vec.z;
      sum[3] += A_val * B_vec.w;
    }

    __syncthreads(); // Synchronize threads to ensure all tiles are processed
  }

  // Write the final partial sums
  if (row < N && base_col < N) {
    // Skip boundary checks for columns (if base_col + 3 < N => safe)
    float *C_out = C + (row * N + base_col);

    C_out[0] = sum[0];
    C_out[1] = sum[1];
    C_out[2] = sum[2];
    C_out[3] = sum[3];
  }
}

int main() {
  // Allocate host memory
  float *h_A = new float[N * N];
  float *h_B = new float[N * N];
  float *h_C = new float[N * N];
  float *h_C_gpu = new float[N * N];

  // Initialize input matrices
  srand(0);
  for (int i = 0; i < N * N; i++) {
    h_A[i] = float(rand() % 10);
    h_B[i] = float(rand() % 10);
    h_C[i] = 0.0f;
    h_C_gpu[i] = 0.0f;
  }

  matrixMulCPU(h_A, h_B, h_C); // CPU baseline

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  size_t size = sizeof(float) * N * N;

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Configure kernel launch
  dim3 block_size(4, 16);
  dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

  matrixMulGPU<<<grid_size, block_size>>>(d_A, d_B, d_C, N); // Launch kernel

  cudaDeviceSynchronize();

  cudaMemcpy(h_C_gpu, d_C, size,
             cudaMemcpyDeviceToHost); // Copy result from device to host

  // Compare results
  long long diff = 0;

  for (int i = 0; i < N; i++) {
    diff += (long long)fabs(h_C_gpu[i] - h_C[i]);
  }

  if (diff == 0)
    std::cout << "Correct!\n";
  else
    std::cout << "Incorrect! diff = " << diff << "\n";

  // Free memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_gpu;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
