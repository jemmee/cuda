// fp64_test.cu
// Simple CUDA demo showcasing FP64 (double-precision) performance on NVIDIA
// GPUs Compares double vs. float matrix multiplication (GEMM) for a square
// matrix Works on any GPU with compute capability >= 1.3 (most do), but fastest
// on those with high FP64 throughput (e.g., A100: ~9.7 TFLOPS FP64; RTX
// 40-series consumer cards: much lower, often 1/64th of FP32)
//
// nvcc fp64_test.cu -o fp64_test
// ./fp64_test

#include <cuda_runtime.h>
#include <stdio.h>

#define N                                                                      \
  4096 // Matrix size (N x N); adjust based on your GPU memory (e.g., 8192 on
       // high-end)
#define THREADS_PER_BLOCK 256

// Simple kernel: C = A * B (naive, no shared memory – educational only)
__global__ void matmul_float(const float *A, const float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
  }
}

__global__ void matmul_double(const double *A, const double *B, double *C,
                              int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
  }
}

int main() {
  size_t size_float = N * N * sizeof(float);
  size_t size_double = N * N * sizeof(double);

  // Host matrices
  float *h_A_f = (float *)malloc(size_float);
  float *h_B_f = (float *)malloc(size_float);
  double *h_A_d = (double *)malloc(size_double);
  double *h_B_d = (double *)malloc(size_double);

  // Initialize with simple values
  for (int i = 0; i < N * N; ++i) {
    h_A_f[i] = 1.0f;
    h_B_f[i] = 2.0f;
    h_A_d[i] = 1.0;
    h_B_d[i] = 2.0;
  }

  // Device matrices
  float *d_A_f, *d_B_f, *d_C_f;
  double *d_A_d, *d_B_d, *d_C_d;

  cudaMalloc(&d_A_f, size_float);
  cudaMalloc(&d_B_f, size_float);
  cudaMalloc(&d_C_f, size_float);
  cudaMalloc(&d_A_d, size_double);
  cudaMalloc(&d_B_d, size_double);
  cudaMalloc(&d_C_d, size_double);

  cudaMemcpy(d_A_f, h_A_f, size_float, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_f, h_B_f, size_float, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_d, h_A_d, size_double, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_d, h_B_d, size_double, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((N + 15) / 16, (N + 15) / 16);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm-up
  matmul_float<<<blocks, threads>>>(d_A_f, d_B_f, d_C_f, N);
  matmul_double<<<blocks, threads>>>(d_A_d, d_B_d, d_C_d, N);
  cudaDeviceSynchronize();

  // FP32 (float) test
  cudaEventRecord(start);
  matmul_float<<<blocks, threads>>>(d_A_f, d_B_f, d_C_f, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms_float;
  cudaEventElapsedTime(&ms_float, start, stop);
  double tflops_float = 2.0 * N * N * N / (ms_float / 1000.0) / 1e12;

  // FP64 (double) test
  cudaEventRecord(start);
  matmul_double<<<blocks, threads>>>(d_A_d, d_B_d, d_C_d, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms_double;
  cudaEventElapsedTime(&ms_double, start, stop);
  double tflops_double = 2.0 * N * N * N / (ms_double / 1000.0) / 1e12;

  printf("Matrix size: %d x %d\n", N, N);
  printf("FP32 (float)  : %.3f ms → %.2f TFLOPS\n", ms_float, tflops_float);
  printf("FP64 (double) : %.3f ms → %.2f TFLOPS\n", ms_double, tflops_double);
  printf("FP64 / FP32 ratio: %.2fx (lower on consumer GPUs)\n",
         tflops_double / tflops_float);

  // Cleanup
  free(h_A_f);
  free(h_B_f);
  free(h_A_d);
  free(h_B_d);
  cudaFree(d_A_f);
  cudaFree(d_B_f);
  cudaFree(d_C_f);
  cudaFree(d_A_d);
  cudaFree(d_B_d);
  cudaFree(d_C_d);

  return 0;
}