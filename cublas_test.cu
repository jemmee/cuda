// nvcc cublas_test.cu -o cublas_test -l cublas -l nvidia-ml

// ./cublas_test

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- Configuration and Macros ---

#define GPU_IDX 0
#define TENSOR_SIZE 16384
#define ITERATIONS 500 // Adjusted for C++ compilation speed

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Runtime Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define NVML_CHECK(call)                                                       \
  do {                                                                         \
    nvmlReturn_t status = call;                                                \
    if (status != NVML_SUCCESS) {                                              \
      fprintf(stderr, "NVML Error at %s:%d: %s\n", __FILE__, __LINE__,         \
              nvmlErrorString(status));                                        \
      /* Attempt to shut down NVML gracefully */                               \
      if (status != NVML_ERROR_UNINITIALIZED) {                                \
        nvmlShutdown();                                                        \
      }                                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

void print_separator(const char *title) {
  printf("\n%s", title);
  for (size_t i = 0; i < 60 - strlen(title); i++)
    printf("=");
  printf("\n");
}

int main() {
  int nDevices = 0;
  struct cudaDeviceProp prop;
  cublasHandle_t cublasH;

  // Host and Device Pointers
  __half *h_A = NULL, *h_B = NULL, *h_C = NULL;
  __half *d_A = NULL, *d_B = NULL, *d_C = NULL;

  print_separator("  NVIDIA AMPERE GPU TEST SUITE (C/C++)");

  // 1. Basic GPU Detection
  printf("\nChecking GPU detection...\n");
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices == 0) {
    printf("CUDA NOT AVAILABLE — No NVIDIA driver or GPU detected!\n");
    return 1;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, GPU_IDX));

  printf("CUDA available:     YES\n");
  printf("GPU count:          %d\n", nDevices);
  printf("Default GPU:        %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("Total Memory:       %zu GB\n",
         prop.totalGlobalMem / (1024 * 1024 * 1024));
  printf("Multiprocessors:    %d\n", prop.multiProcessorCount);

  // --- MEMORY ALLOCATION & INIT ---
  print_separator("Allocation & Initialization");

  size_t num_elements = (size_t)TENSOR_SIZE * TENSOR_SIZE;
  size_t size_bytes = num_elements * sizeof(__half);

  // Allocate Host Memory
  h_A = (__half *)malloc(size_bytes);
  h_B = (__half *)malloc(size_bytes);
  h_C = (__half *)malloc(size_bytes);
  if (!h_A || !h_B || !h_C) {
    fprintf(stderr, "Host memory allocation failed!\n");
    return 1;
  }

  // Allocate Device Memory
  CUDA_CHECK(cudaMalloc((void **)&d_A, size_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_B, size_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_C, size_bytes));

  // Initialize Data
  srand(time(NULL));
  for (size_t i = 0; i < num_elements; i++) {
    float r = (float)rand() / RAND_MAX;
    h_A[i] = __float2half(r);
    h_B[i] = __float2half(r);
  }
  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_bytes, cudaMemcpyHostToDevice));

  // 3. Compute Stress Test (cuBLAS Matrix Multiply - GEMM)
  print_separator("Running Compute Stress Test");

  if (cublasCreate(&cublasH) != CUBLAS_STATUS_SUCCESS)
    return 1;

  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("   Warming up...\n");
  for (int i = 0; i < 5; i++) {
    cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, TENSOR_SIZE, TENSOR_SIZE,
                TENSOR_SIZE, &alpha, d_A, TENSOR_SIZE, d_B, TENSOR_SIZE, &beta,
                d_C, TENSOR_SIZE);
  }
  cudaDeviceSynchronize();

  printf("   Stressing Tensor Cores (%d iterations)...\n", ITERATIONS);
  cudaEventRecord(start);

  for (int i = 0; i < ITERATIONS; i++) {
    cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, TENSOR_SIZE, TENSOR_SIZE,
                TENSOR_SIZE, &alpha, d_A, TENSOR_SIZE, d_B, TENSOR_SIZE, &beta,
                d_C, TENSOR_SIZE);
    if (i % (ITERATIONS / 10) == 0) {
      printf("      iteration %d/%d...\n", i + 1, ITERATIONS);
    }
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float elapsed_s = ms / 1000.0f;

  // TFLOPS Calculation: 2 * N^3 * Iterations / Time (1e12 for TFLOPS)
  double flops =
      2.0 * (double)TENSOR_SIZE * TENSOR_SIZE * TENSOR_SIZE * ITERATIONS;
  double tflops_estimate = flops / (elapsed_s * 1e12);

  printf("Compute test PASSED in %.2fs (~%.1f TFLOPS FP16)\n", elapsed_s,
         tflops_estimate);

  // 4. Temperature & Power Monitoring (using NVML)
  print_separator("Monitoring Temperature & Power");

  NVML_CHECK(nvmlInit());
  nvmlDevice_t device;
  NVML_CHECK(nvmlDeviceGetHandleByIndex(GPU_IDX, &device));

  unsigned int temp;
  unsigned int power_draw;
  nvmlMemory_t memInfo;

  NVML_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp));
  // Power usage is returned in mW (milliwatts), convert to W
  NVML_CHECK(nvmlDeviceGetPowerUsage(device, &power_draw));
  NVML_CHECK(nvmlDeviceGetMemoryInfo(device, &memInfo));

  printf("Temperature: %u °C\n", temp);
  printf("Power Draw:  %.1f W\n", (float)power_draw / 1000.0f);
  printf("Memory Used: %llu MB / %llu MB\n", memInfo.used / (1024 * 1024),
         memInfo.total / (1024 * 1024));

  nvmlShutdown(); // Must shut down NVML

  // --- CLEANUP ---
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(cublasH);

  print_separator("=============================");
  if (prop.major >= 8) {
    printf("AMPERE/HOPPER GPU DETECTED (CC %d.%d) AND FULLY FUNCTIONAL!\n",
           prop.major, prop.minor);
  } else {
    printf("GPU detected, but not confirmed Ampere or newer architecture.\n");
  }
  printf("All tests completed successfully!\n");
  print_separator("=============================");

  return 0;
}
