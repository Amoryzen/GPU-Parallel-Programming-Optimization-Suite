#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMulGPU(float *a, float *b, float *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
      sum += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
  }
}

void matrixMulCPU(const float *a, const float *b, float *c, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;

      for (int k = 0; k < N; k++) {
        sum += a[i * N + k] * b[k * N + j];
      }

      c[i * N + j] = sum;
    }
  }
}

int main() {
  const int N = 512;

  float *h_a = new float[N * N];
  float *h_b = new float[N * N];
  float *h_c = new float[N * N];
  float *h_c_gpu = new float[N * N];

  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, sizeof(float) * N * N);
  cudaMalloc((void **)&d_b, sizeof(float) * N * N);
  cudaMalloc((void **)&d_c, sizeof(float) * N * N);

  for (int i = 0; i < N; i++) {
    h_a[i] = std::rand() % 10;
    h_b[i] = std::rand() % 10;
    h_c[i] = 0;
    h_c_gpu[i] = 0;
  }

  matrixMulCPU(h_a, h_b, h_c, N);

  cudaMemcpy(d_a, h_a, sizeof(float) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(float) * N * N, cudaMemcpyHostToDevice);

  int block_size = 16;
  dim3 threads_per_block(block_size, block_size);
  dim3 grid_size((N + block_size - 1) / block_size,
                 (N + block_size - 1) / block_size);

  matrixMulGPU<<<grid_size, threads_per_block>>>(d_a, d_b, d_c, N);

  cudaMemcpy(h_c_gpu, d_c, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

  long temp = 0;

  for (int i = 0; i < N; i++) {
    temp += std::abs(h_c_gpu[i] - h_c[i]);
  }

  if (temp == 0)
    std::cout << "Correct!\n";
  else
    std::cout << "Incorrect\n";

  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  delete[] h_c_gpu;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
