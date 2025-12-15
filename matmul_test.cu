// matmul_test.cu
// CUDA program to compare performance: Naive vs. Tiled vs. cuBLAS matrix
// multiplication (SGEMM)
//
// Compile with: nvcc matmul_test.cu -o matmul_test -l cublas
// Run with: ./matmul_test

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32 // Good choice for most modern GPUs

// --------------------- Naive Kernel ---------------------
__global__ void naiveMatmul(float *A, float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// --------------------- Tiled Kernel (Shared Memory) ---------------------
__global__ void tiledMatmul(float *A, float *B, float *C, int N) {
  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  float sum = 0.0f;

  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    if (row < N && (t * TILE_SIZE + tx) < N)
      sA[ty][tx] = A[row * N + t * TILE_SIZE + tx];
    else
      sA[ty][tx] = 0.0f;

    if (col < N && (t * TILE_SIZE + ty) < N)
      sB[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
    else
      sB[ty][tx] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += sA[ty][k] * sB[k][tx];
    }
    __syncthreads();
  }

  if (row < N && col < N) {
    C[row * N + col] = sum;
  }
}

// --------------------- Helper Functions ---------------------
void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void checkCublasStatus(cublasStatus_t stat, const char *msg) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s: cuBLAS error\n", msg);
    exit(EXIT_FAILURE);
  }
}

float runKernel(void kernel(float *, float *, float *, int), dim3 blocks,
                dim3 threads, float *d_A, float *d_B, float *d_C, int N) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm-up
  kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return ms;
}

// --------------------- Main ---------------------
int main() {
  int N = 16384;

  size_t size = N * N * sizeof(float);

  // Host arrays (initialize with simple values for verification)
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  for (int i = 0; i < N * N; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // Device arrays
  float *d_A, *d_B, *d_C;
  checkCudaError(cudaMalloc(&d_A, size), "Malloc A");
  checkCudaError(cudaMalloc(&d_B, size), "Malloc B");
  checkCudaError(cudaMalloc(&d_C, size), "Malloc C");

  checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Copy A");
  checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Copy B");

  // Grid/Block setup
  dim3 threadsPerBlock(16, 16);
  dim3 blocksNaive((N + 15) / 16, (N + 15) / 16);
  dim3 threadsTiled(TILE_SIZE, TILE_SIZE);
  dim3 blocksTiled((N + TILE_SIZE - 1) / TILE_SIZE,
                   (N + TILE_SIZE - 1) / TILE_SIZE);

  // --------------------- Run Naive ---------------------
  float time_naive =
      runKernel(naiveMatmul, blocksNaive, threadsPerBlock, d_A, d_B, d_C, N);
  double gflops_naive = 2.0 * N * N * N / (time_naive / 1000.0) / 1e12;

  // --------------------- Run Tiled ---------------------
  float time_tiled =
      runKernel(tiledMatmul, blocksTiled, threadsTiled, d_A, d_B, d_C, N);
  double gflops_tiled = 2.0 * N * N * N / (time_tiled / 1000.0) / 1e12;

  // --------------------- Run cuBLAS ---------------------
  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle), "cuBLAS init");

  float alpha = 1.0f, beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm-up
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N,
              &beta, d_C, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_cublas = 0;
  cudaEventElapsedTime(&time_cublas, start, stop);
  double gflops_cublas = 2.0 * N * N * N / (time_cublas / 1000.0) / 1e12;

  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // --------------------- Results ---------------------
  printf("Matrix size: %d x %d\n\n", N, N);
  printf("Naive Kernel      : %.3f ms   →   %.2f GFLOPS\n", time_naive,
         gflops_naive);
  printf("Tiled Kernel      : %.3f ms   →   %.2f GFLOPS\n", time_tiled,
         gflops_tiled);
  printf("cuBLAS (SGEMM)    : %.3f ms   →   %.2f GFLOPS\n\n", time_cublas,
         gflops_cublas);

  printf("Speedup (Tiled vs Naive)   : %.1fx\n", time_naive / time_tiled);
  printf("Speedup (cuBLAS vs Naive)  : %.1fx\n", time_naive / time_cublas);
  printf("Speedup (cuBLAS vs Tiled)  : %.1fx\n", time_tiled / time_cublas);

  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}