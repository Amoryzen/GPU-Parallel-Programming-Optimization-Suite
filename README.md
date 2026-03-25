# GPU Parallel Programming Optimization Suite

A comprehensive collection of CUDA C++ examples demonstrating fundamental to advanced GPU acceleration techniques, focusing on memory management, kernel optimization, and high-performance libraries.

## Problem Statement and Motivation

Modern applications in AI, scientific computing, and rendering require massive computational power that traditional CPUs cannot provide. GPU parallel programming utilizing NVIDIA's CUDA platform bridges this gap. This project serves as a hands-on guide and reference implementation for learning how to harvest the full potential of GPUs—progressing from basic vector operations to highly optimized matrix multiplications using Tensor Cores, cuBLAS, and CUTLASS.

## Key Features

- **Vector Operations:**
  - Standard Vector Addition
  - Advanced Vector Addition utilizing chunks and CUDA streams
  - Parallel Vector Reduction strategies
- **Matrix Operations:**
  - Matrix Addition (optimized for full theoretical occupancy)
  - Naive Matrix Multiplication
  - High-performance Matrix Multiplication using cuBLAS (`matrixMulcuBLASS.cu`)
  - Hardware-accelerated Matrix Multiplication using Tensor Cores / WMMA API (`matrixMulWMMA.cu`)
  - State-of-the-art Matrix Multiplication using the CUTLASS library (`matrixMulcuTLASS.cu`)
- **CUDA API & Profiling:**
  - CUDA Runtime API usage examples
  - Extensive profiling data logs generated with Nsight Compute and Nsight Systems

## Tech Stack and Dependencies

- **Hardware:** NVIDIA GPU (Compute Capability 7.0+ recommended for Tensor Core / WMMA features)
- **Software:** 
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (`nvcc` compiler)
  - [NVIDIA Nsight Compute & Nsight Systems](https://developer.nvidia.com/tools-overview) (for analyzing the provided `.ncu-rep` and `.nsys-rep` files)
- **Libraries:**
  - cuBLAS (included with CUDA Toolkit)
  - [CUTLASS](https://github.com/NVIDIA/cutlass) (included as a submodule)

## Project Structure

```text
.
├── arc/                  # Additional scripts and resources (e.g., WSL setup scripts)
├── bin/                  # Compiled executable binaries
├── external/             # External dependencies (contains the CUTLASS submodule)
├── include/              # Header files
├── profiling/            # Nsight Compute and Nsight Systems profiling reports
└── src/                  # CUDA source code (.cu examples)
```

## Installation and Setup

1. **Clone the repository:**
   Ensure you clone with the `--recursive` flag to pull in external dependencies like CUTLASS.
   ```bash
   git clone --recursive https://github.com/yourusername/Mastering-GPU-Parallel-Programming-with-CUDA.git
   cd Mastering-GPU-Parallel-Programming-with-CUDA
   ```

2. **Verify CUDA Installation:**
   ```bash
   nvcc --version
   ```

3. **Submodule Updates (Optional):**
   If you cloned without `--recursive`, initialize the submodules manually:
   ```bash
   git submodule update --init --recursive
   ```

## Usage Examples

Compile individual source files using the `nvcc` compiler. Ensure you link against required libraries (e.g., `-lcublas`) depending on the execution target.

**Example 1: Basic Vector Addition**
```bash
nvcc -O3 src/vectorAdd.cu -o bin/vectorAdd
./bin/vectorAdd
```

**Example 2: Compiling cuBLAS Matrix Multiplication**
```bash
nvcc -O3 src/matrixMulcuBLASS.cu -lcublas -o bin/matrixMulcuBLASS
./bin/matrixMulcuBLASS
```

**Example 3: Compiling with CUTLASS**
When compiling CUTLASS files, ensure the include paths to the submodule are specified:
```bash
nvcc -O3 -I external/cutlass/include src/matrixMulcuTLASS.cu -o bin/matrixMulcuTLASS
./bin/matrixMulcuTLASS
```

## Configuration Options

- **Runtime Prompts:** Certain advanced kernels (e.g., `matrixMulWMMA.cu`) have been designed with interactive runtime prompts, allowing you to dynamically configure execution dimension variables such as matrix size (`N`) and `TILE_SIZE`.

## Testing Instructions

Validation logic is built into each source file. After computing the results on the device (GPU), the data is transferred back to the host (CPU) and compared against a standard CPU implementation.
Run any compiled executable; a successful run will typically output `Success!` or indicate a `Max error: 0.0`.

## Performance Notes and Limitations

- **Profiling:** The `profiling/` directory contains various `.ncu-rep` and `.nsys-rep` reports. These can be opened with NVIDIA Nsight tools to view occupancy, warp divergence, and memory throughput metrics.
- **Hardware Limitations:** Executing Tensor Core (WMMA) examples on older GPUs (Volta architecture or older) will fail as the hardware does not support these instructions.
- **Memory Bandwidth:** Naive implementations are heavily memory-bound. Review the cuBLAS and CUTLASS implementations to see how shared memory tiling optimally skirts these bottlenecks.

## Roadmap / Future Work

- Implement parallel prefix sum (Scan) algorithms.
- Develop custom shared memory tiled matrix multiplication to bridge the gap between naive and cuBLAS performance.
- Expand multi-GPU standard implementations.
- Add CMake configurations for streamlined cross-platform builds.

## Contributing Guidelines

Contributions are welcome! If you have a highly optimized kernel or an example of a new CUDA feature:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingKernel`).
3. Ensure your file contains host-side numerical validation.
4. Commit your changes (`git commit -m 'Add AmazingKernel'`).
5. Push to the branch (`git push origin feature/AmazingKernel`).
6. Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. *(Note: Assumed standard open-source license, please add a LICENSE file if one does not exist).*
