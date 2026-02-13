#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>


#define N 1024       // size of matrix
#define TILE_SIZE 16 // size of tile

// GPU Kernel for matrix multiplication (assuming N % TILE_SIZE == 0)
__global__ void matrixMulGPU(float *a, float *b, float *c) {
  // Declare shared memory for tiles
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  // Calculate global row and column
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f; // Initialize sum for this thread

  int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE; // Number of tiles to process

  for (int t = 0; t < num_tiles; t++) {
    // Load tileA and tileB from global memory to shared memory
    tileA[threadIdx.y][threadIdx.x] =
        a[row * N + (t * TILE_SIZE + threadIdx.x)];
    tileB[threadIdx.y][threadIdx.x] =
        b[(t * TILE_SIZE + threadIdx.y) * N + col];

    __syncthreads(); // Synchronize threads to ensure all tiles are loaded

    // Perform matrix multiplication
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads(); // Synchronize threads to ensure all tiles are processed
  }

  c[row * N + col] = sum; // Store the result in global memory
}

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

int main() {
  // Allocate host memory
  float *h_A = new float[N * N];
  float *h_B = new float[N * N];
  float *h_C = new float[N * N];
  float *h_C_gpu = new float[N * N];

  // Initialize input matrices
  for (int i = 0; i < N * N; i++) {
    h_A[i] = static_cast<float>(i % 10);       // random values
    h_B[i] = static_cast<float>((i * 2) % 10); // random values
  }

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float) * N * N);
  cudaMalloc(&d_B, sizeof(float) * N * N);
  cudaMalloc(&d_C, sizeof(float) * N * N);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

  dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
  dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

  // Launch kernel
  matrixMulCPU(h_A, h_B, h_C);
  matrixMulGPU<<<grid_size, threads_per_block>>>(d_A, d_B, d_C);

  cudaMemcpy(h_C_gpu, d_C, sizeof(float) * N * N,
             cudaMemcpyDeviceToHost); // Copy result from device to host

  // Compare results
  long temp = 0;

  for (int i = 0; i < N; i++) {
    temp += std::abs(h_C_gpu[i] - h_C[i]);
  }

  if (temp == 0)
    std::cout << "Correct!\n";
  else
    std::cout << "Incorrect\n";

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
