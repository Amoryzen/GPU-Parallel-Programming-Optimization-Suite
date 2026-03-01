#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>


// Essential arguments N, alpha, beta will be retrieved from user input

// Macro for checking CUDA errors
#define cudaCheckError(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPU assert: " << cudaGetErrorString(code) << " " << file
              << " " << line << std::endl;

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
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << " "
              << file << " " << line << std::endl;

    if (abort)
      exit(err);
  }
}

// CPU Matrix Multiplication (Reference)
void cpu_matrix_multiply(half *A, half *B, float *C, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;

      for (int k = 0; k < n; k++) {
        sum += __half2float(A[i * n + k]) * __half2float(B[k * n + j]);
      }

      C[i * n + j] = sum;
    }
  }
}

int main() {
  int n;
  float alpha, beta;

  std::cout << "Enter N: ";
  if (std::cin >> n) {
    std::cerr << "Error reading N.\n";
    return 1;
  }

  std::cout << "Enter alpha: ";
  if (std::cin >> alpha) {
    std::cerr << "Error reading alpha.\n";
    return 1;
  }

  std::cout << "Enter beta: ";
  if (std::cin >> beta) {
    std::cerr << "Error reading beta.\n";
    return 1;
  }

  // Initialize cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Allocate memory on the host
  size_t total_elements = n * n;
  half *h_A = new half[total_elements];
  half *h_B = new half[total_elements];
  float *h_C = new float[total_elements];
  float *h_C_CPU = new float[total_elements];

  // Initialize matrices with random values
  for (int i = 0; i < total_elements; i++) {
    h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
  }

  // Allocate memory on the device
  half *d_A, *d_B;
  float *d_C;

  // Allocate memory on the device
  cudaCheckError(cudaMalloc(&d_A, total_elements * sizeof(half)));
  cudaCheckError(cudaMalloc(&d_B, total_elements * sizeof(half)));
  cudaCheckError(cudaMalloc(&d_C, total_elements * sizeof(float)));

  // Copy data from host to device
  cudaCheckError(cudaMemcpy(d_A, h_A, total_elements * sizeof(half),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, h_B, total_elements * sizeof(half),
                            cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  cudaCheckError(cudaEventRecord(start)); // Start recording

  // Perform matrix multiplication
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A,
               CUDA_R_16F, n, d_B, CUDA_R_16F, n, &beta, d_C, CUDA_R_32F, n,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  gpuKernelCheck();

  cudaCheckError(cudaEventRecord(stop));      // Stop recording
  cudaCheckError(cudaEventSynchronize(stop)); // Synchronize event

  // Calculate and print the execution time
  float milliseconds = 0;
  cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "Time elapsed: " << milliseconds << " ms\n";

  // Copy result from device to host
  cudaCheckError(cudaMemcpy(h_C, d_C, total_elements * sizeof(float),
                            cudaMemcpyDeviceToHost));

  cpu_matrix_multiply(h_A, h_B, h_C_CPU,
                      n); // CPU Matrix Multiplication (Reference)

  // Print result
  std::cout << "First 10 elements of GPU result: \n";
  for (int i = 0; i < 10; i++) {
    std::cout << h_C[i] << " ";
  }
  std::cout << "\n";

  std::cout << "First 10 elements of CPU result: \n";
  for (int i = 0; i < 10; i++) {
    std::cout << h_C_CPU[i] << " ";
  }
  std::cout << "\n";

  // Free memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_CPU;

  cudaCheckError(cudaFree(d_A));
  cudaCheckError(cudaFree(d_B));
  cudaCheckError(cudaFree(d_C));

  cublasDestroy(handle); // Destroy cuBLAS handle

  return 0;
}
