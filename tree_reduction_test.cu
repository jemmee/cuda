// nvcc tree_reduction_test.cu -o tree_reduction_test
//
// ./tree_reduction_test

// tree_reduction_demo.cu
// CUDA demo: Parallel sum reduction using a tree-based (binary tree) approach
// Reduces a large array of floats to a single sum using multiple kernel phases
// Highly efficient – one of the classic CUDA parallel patterns

#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For __syncthreads()
#include <stdio.h>

#define N (1 << 24) // 16 million elements (~64 MB)
#define THREADS_PER_BLOCK 256
#define BLOCKS (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK

// Kernel: Tree-based reduction within each block
// Each block reduces its portion into shared memory and writes one result per
// block
__global__ void tree_reduce_kernel(float *in, float *out, int n) {
  extern __shared__ float sdata[]; // Shared memory size set at launch

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory (coalesced)
  float val = (idx < n) ? in[idx] : 0.0f;
  sdata[tid] = val;
  __syncthreads();

  // Tree reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Thread 0 in each block writes partial sum to global output
  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

int main() {
  printf("Tree Reduction Demo: Sum of %d elements\n", N);

  size_t bytes = N * sizeof(float);

  // Host and device arrays
  float *h_in = (float *)malloc(bytes);
  float *h_out = (float *)malloc(BLOCKS * sizeof(float));

  float *d_in, *d_out;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, BLOCKS * sizeof(float));

  // Initialize input with 1.0f (exact sum should be N)
  for (int i = 0; i < N; ++i) {
    h_in[i] = 1.0f;
  }
  cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

  // Shared memory size: one float per thread
  size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(float);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // First level: reduce to one value per block
  cudaEventRecord(start);
  tree_reduce_kernel<<<BLOCKS, THREADS_PER_BLOCK, shared_mem_size>>>(d_in,
                                                                     d_out, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  // Final reduction on CPU (or launch a second kernel if BLOCKS > 1)
  cudaMemcpy(h_out, d_out, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

  float gpu_sum = 0.0f;
  for (int i = 0; i < BLOCKS; ++i) {
    gpu_sum += h_out[i];
  }

  printf("GPU reduction time: %.3f ms\n", ms);
  printf("GPU sum: %.1f\n", gpu_sum);
  printf("Expected sum: %.1f\n", (float)N);
  printf("Error: %.1f\n", gpu_sum - (float)N);

  // Cleanup
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}