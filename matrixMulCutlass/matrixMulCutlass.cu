// Include standard libraries
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/gemm/device/gemm.h" // NOLINT Include CUTLASS GEMM device code

// Define a CUTLASS GEMM template and launch a GEMM kernel
cudaError_t CUTLASS_GEMM(int M, int N, int K, float alpha, float *const A,
                         int lda, float *const B, int ldb, float beta, float *C,
                         int ldc) {
  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 127x128x8 threadblock tile size
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,        // Data type of matrix A
                                  ColumnMajor,  // Layout of matrix A
                                  float,        // Data type of matrix B
                                  ColumnMajor,  // Layout of matrix B
                                  float,        // Data type of matrix C
                                  ColumnMajor>; // Layout of matrix C

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object
  CutlassGemm::Arguments args({M, N, K},      // GEMM problem dimensions
                              {A, lda},       // Matrix A descriptor
                              {B, ldb},       // Matrix B descriptor
                              {C, ldc},       // Matrix C descriptor
                              {C, ldc},       // Matrix D descriptor
                              {alpha, beta}); // Scalar values

  cutlass::Status status = gemm_operator(args); // Launch the CUTLAS GEMM kernel

  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess; // Return success if the CUTLASS GEMM operator returned
                      // successfully
}

// Kernel to initialize a matrix with small integers
__global__ void InitMatrix(float *matrix, int rows, int columns, int seed = 0) {
  // Calculate the global row and column indices for the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if the current thread is within the bounds of the matrix
  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

// Simple function to initialize a matrix to arbitrary small integers
cudaError_t InitMatrixOnDevice(float *matrix, int rows, int columns,
                               int seed = 0) {
  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);

  InitMatrix<<<grid, block>>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

// Allocate device memory for a matrix then fill with arbitrary small integers
cudaError_t AllocateMatrix(float **matrix, int rows, int columns,
                           int seed = 0) {
  cudaError_t result;

  size_t size_of_matrix = sizeof(float) * rows * columns;

  result = cudaMalloc(reinterpret_cast<void **>(matrix),
                      size_of_matrix); // Allocate device memory

  // Check if the allocation was successful
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result)
              << std::endl;

    return result;
  }

  result = cudaMemset(*matrix, 0, size_of_matrix); // Clear the allocation

  // Check if the clearing was successful
  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
              << cudaGetErrorString(result) << std::endl;

    return result;
  }

  result = InitMatrixOnDevice(
      *matrix, rows, columns,
      seed); // Initialize the matrix with arbitrary small integers

  // Check if the initialization was successful
  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: " << cudaGetErrorString(result)
              << std::endl;

    return result;
  }

  return result;
}

// Naive referennce GEMM computation
__global__ void NaiveGEMM(int M, int N, int K, float alpha, float *const A,
                          int lda, float *const B, int ldb, float beta,
                          float *C, int ldc) {
  // Calculate the global row and column indices for the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if the current thread is within the bounds of the matrix
  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; k++) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

// Naive reference GEMM computation
cudaError_t NaiveGEMMOnDevice(int M, int N, int K, float alpha, float *const A,
                              int lda, float *const B, int ldb, float beta,
                              float *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  NaiveGEMM<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

// Allocate several matrices in the GPU device memory and call a
// single-precision CUTLASS GEMM kernel
cudaError_t TestCUTLASSGEMM(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  // Define several matrices to be used as operands to the GEMM kernels

  // Compute leading dimensions for each matrix
  int lda = M;
  int ldb = K;
  int ldc = M;

  size_t size_of_C =
      sizeof(float) * ldc * N; // Compute size in bytes of the C matrix

  // Allocate several matrices in the GPU device memory
  float *A;
  float *B;
  float *C_cutlass;
  float *C_naive;

  // Allocate matrices in the GPU device memory with arbitrary seeds
  result = AllocateMatrix(&A, M, K);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix A: " << cudaGetErrorString(result)
              << std::endl;

    return result;
  }

  result = AllocateMatrix(&B, K, N);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix B: " << cudaGetErrorString(result)
              << std::endl;

    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix C_cutlass: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(A);
    cudaFree(B);

    return result;
  }

  result = AllocateMatrix(&C_naive, M, N);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix C_naive: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);

    return result;
  }

  result = cudaMemcpy(C_naive, C_cutlass, size_of_C, cudaMemcpyDeviceToDevice);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy matrix C_naive: " << cudaGetErrorString(result)
              << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    cudaFree(C_naive);

    return result;
  }

  // Call a single-precision CUTLASS GEMM kernel
  result = CUTLASS_GEMM(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Failed to call CUTLASS GEMM kernel: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    cudaFree(C_naive);

    return result;
  }

  // Verify

  result =
      NaiveGEMMOnDevice(M, N, K, alpha, A, lda, B, ldb, beta, C_naive, ldc);
  if (result != cudaSuccess) {
    std::cerr << "Failed to call Naive GEMM kernel: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    cudaFree(C_naive);

    return result;
  }

  // Copy to host and veerify equivalence
  std::vector<float> host_cutlass(ldc * N, 0);
  std::vector<float> host_naive(ldc * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, size_of_C,
                      cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to host: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    cudaFree(C_naive);

    return result;
  }

  result =
      cudaMemcpy(host_naive.data(), C_naive, size_of_C, cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_naive matrix to host: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    cudaFree(C_naive);

    return result;
  }

  if (host_cutlass != host_naive) {
    std::cerr << "CUTLASS GEMM results do not match naive GEMM results!"
              << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    cudaFree(C_naive);

    return cudaErrorInvalidValue;
  }

  // Free device memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C_cutlass);
  cudaFree(C_naive);

  return result;
}

int main(int argc, const char *argv[]) {
  // Parse the command line arguments to obtain GEMM dimensions and scalar
  // values
  int problem[3] = {1024, 1024, 1024};

  for (int i = 1; i < argc && i < 4; i++) {
    std::stringstream ss(argv[i]);
    ss >> problem[i - 1];
  }

  float scalars[2] = {
      1, 0}; // Scalars used for linear scaling the result of the matrix product

  for (int i = 4; i < argc && i < 6; i++) {
    std::stringstream ss(argv[i]);
    ss >> scalars[i - 4];
  }

  // Run the CUTLASS GEMM test
  cudaError_t result = TestCUTLASSGEMM(problem[0], // GEMM M dimension
                                       problem[1], // GEMM N dimension
                                       problem[2], // GEMM K dimension
                                       scalars[0], // GEMM alpha scalar
                                       scalars[1]  // GEMM beta scalar
  );

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM test failed: " << cudaGetErrorString(result)
              << std::endl;

    return 1;
  }

  if (result == cudaSuccess) {
    std::cout << "CUTLASS GEMM test PASSED" << std::endl;
  }

  return result == cudaSuccess ? 0 : 1;
}
