// fft_test.cu
// Simple CUDA program to compute 1D FFT using:
// 1. cuFFT (NVIDIA's highly optimized FFT library)
// 2. A very basic naive radix-2 FFT kernel (for educational comparison only)
//
// This demonstrates the massive performance difference between cuFFT and a
// naive implementation.
//
// nvcc fft_test.cu -o fft_test -l cufft
// ./ fft_test

#include <chrono>
#include <cmath>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <vector>

// Size of the signal (Power of 2 is best for FFT)
// WARNING: If you go > 16384, the Naive O(N^2) kernel will be extremely slow.
const int N = 16384;
const size_t BYTES = N * sizeof(cufftComplex);

// #define M_PI 3.14159265358979323846f

// Error checking macro
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// ----------------------------------------------------------------------
// 1. NAIVE DFT KERNEL O(N^2)
// Formula: X[k] = sum_{n=0}^{N-1} x[n] * exp(-i * 2 * pi * k * n / N)
// ----------------------------------------------------------------------
__global__ void naive_dft_kernel(const cufftComplex *input,
                                 cufftComplex *output, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < n) {
    float sum_r = 0.0f;
    float sum_i = 0.0f;

    for (int t = 0; t < n; ++t) {
      float angle = -2.0f * M_PI * k * t / n;
      float c = cosf(angle);
      float s = sinf(angle);

      // Complex Multiply: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
      // Input sample: input[t].x + i*input[t].y
      // Euler term:   c + i*s
      float in_r = input[t].x;
      float in_i = input[t].y;

      sum_r += (in_r * c - in_i * s);
      sum_i += (in_r * s + in_i * c);
    }
    output[k].x = sum_r;
    output[k].y = sum_i;
  }
}

// ----------------------------------------------------------------------
// UTILS
// ----------------------------------------------------------------------
void init_signal(cufftComplex *data, int n) {
  for (int i = 0; i < n; ++i) {
    // Generate a test signal: sum of two sine waves
    float t = (float)i / n;
    float val = sinf(2 * M_PI * 5.0f * t) + 0.5f * sinf(2 * M_PI * 10.0f * t);
    data[i].x = val;  // Real part
    data[i].y = 0.0f; // Imaginary part
  }
}

// Compare two complex arrays (Root Mean Square Error)
bool verify_result(const cufftComplex *ref, const cufftComplex *test, int n) {
  float error_sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    float diff_r = ref[i].x - test[i].x;
    float diff_i = ref[i].y - test[i].y;
    error_sum += diff_r * diff_r + diff_i * diff_i;
  }
  float rmse = sqrtf(error_sum / n);

  std::cout << ">> RMSE Error: " << rmse << std::endl;
  // Allow small error due to floating point accumulation differences
  return rmse < 1e-3;
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------
int main() {
  std::cout << "FFT Comparison Test (Size N=" << N << ")" << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  // Host memory
  cufftComplex *h_signal = (cufftComplex *)malloc(BYTES);
  cufftComplex *h_out_naive = (cufftComplex *)malloc(BYTES);
  cufftComplex *h_out_cufft = (cufftComplex *)malloc(BYTES);

  init_signal(h_signal, N);

  // Device memory
  cufftComplex *d_signal, *d_out_naive, *d_out_cufft;
  CHECK_CUDA(cudaMalloc(&d_signal, BYTES));
  CHECK_CUDA(cudaMalloc(&d_out_naive, BYTES));
  CHECK_CUDA(cudaMalloc(&d_out_cufft, BYTES));

  // Copy input to device
  CHECK_CUDA(cudaMemcpy(d_signal, h_signal, BYTES, cudaMemcpyHostToDevice));

  // -------------------------------------------------------
  // Benchmark 1: Naive DFT Kernel
  // -------------------------------------------------------
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  std::cout << "Running Naive DFT Kernel (O(N^2))... ";
  CHECK_CUDA(cudaDeviceSynchronize());
  auto start_naive = std::chrono::high_resolution_clock::now();

  naive_dft_kernel<<<numBlocks, blockSize>>>(d_signal, d_out_naive, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end_naive = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> naive_ms = end_naive - start_naive;
  std::cout << "Done. Time: " << naive_ms.count() << " ms" << std::endl;

  // -------------------------------------------------------
  // Benchmark 2: cuFFT (Fast Fourier Transform)
  // -------------------------------------------------------
  cufftHandle plan;
  // Create a 1D FFT plan for Complex-to-Complex (C2C)
  if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
    std::cerr << "cuFFT Plan creation failed!" << std::endl;
    return 1;
  }

  std::cout << "Running cuFFT (O(N log N))...        ";
  CHECK_CUDA(cudaDeviceSynchronize());
  auto start_cufft = std::chrono::high_resolution_clock::now();

  // Execute FFT
  // Note: cuFFT can be in-place or out-of-place. We use out-of-place here.
  if (cufftExecC2C(plan, d_signal, d_out_cufft, CUFFT_FORWARD) !=
      CUFFT_SUCCESS) {
    std::cerr << "cuFFT Execution failed!" << std::endl;
    return 1;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end_cufft = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cufft_ms = end_cufft - start_cufft;
  std::cout << "Done. Time: " << cufft_ms.count() << " ms" << std::endl;

  // -------------------------------------------------------
  // Verification
  // -------------------------------------------------------
  CHECK_CUDA(
      cudaMemcpy(h_out_naive, d_out_naive, BYTES, cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(h_out_cufft, d_out_cufft, BYTES, cudaMemcpyDeviceToHost));

  // Verify Naive vs cuFFT (Treating cuFFT as the gold standard)
  if (verify_result(h_out_cufft, h_out_naive, N)) {
    std::cout << ">> Results MATCH!" << std::endl;
  } else {
    std::cout << ">> Results MISMATCH!" << std::endl;
  }

  std::cout << "\n------------------------------------------" << std::endl;
  std::cout << "Speedup: " << naive_ms.count() / cufft_ms.count() << "x"
            << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  // Cleanup
  cufftDestroy(plan);
  cudaFree(d_signal);
  cudaFree(d_out_naive);
  cudaFree(d_out_cufft);
  free(h_signal);
  free(h_out_naive);
  free(h_out_cufft);

  return 0;
}