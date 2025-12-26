// mersenne_primes_test.cu
// Find Mersenne primes (2^p - 1) using CUDA with Lucas-Lehmer test
// Each thread tests one odd exponent p independently
// Limited to p <= 63 due to 64-bit integer overflow (2^64 - 1 max)
// Compile: nvcc mersenne_primes_test.cu -o mersenne_primes_test
// Run: ./mersenne_primes_test

#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_P 1000000000ULL // Upper limit for p (odd numbers up to this)
#define THREADS_PER_BLOCK 256

__global__ void lucas_lehmer_kernel(unsigned int *exponents,
                                    unsigned char *results, int num_exponents) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_exponents)
    return;

  unsigned int p = exponents[idx];
  if (p < 3) {
    results[idx] = 0;
    return;
  }

  // For p > 63, 2^p - 1 exceeds uint64_t → skip (or use big-int for larger p)
  if (p > 63) {
    results[idx] = 0;
    return;
  }

  unsigned long long M = (1ULL << p) - 1; // 2^p - 1
  unsigned long long s = 4;

  // Lucas-Lehmer iteration: s = (s*s - 2) % M, repeat p-2 times
  for (int i = 0; i < p - 2; ++i) {
    s = (s * s - 2) % M;
  }

  results[idx] = (s == 0) ? 1 : 0;
}

int main() {
  printf("Searching for Mersenne primes with odd p from 3 to %llu...\n\n",
         MAX_P);

  // Count odd numbers from 3 to MAX_P
  int num_odd = (MAX_P - 1) / 2;
  size_t exponents_size = num_odd * sizeof(unsigned int);
  size_t results_size = num_odd * sizeof(unsigned char);

  // Host arrays
  unsigned int *h_exponents = (unsigned int *)malloc(exponents_size);
  unsigned char *h_results = (unsigned char *)malloc(results_size);

  // Fill odd exponents: 3,5,7,...,MAX_P
  for (int i = 0; i < num_odd; ++i) {
    h_exponents[i] = 3 + 2 * i;
  }

  // Device arrays
  unsigned int *d_exponents;
  unsigned char *d_results;
  cudaMalloc(&d_exponents, exponents_size);
  cudaMalloc(&d_results, results_size);

  cudaMemcpy(d_exponents, h_exponents, exponents_size, cudaMemcpyHostToDevice);

  // Launch kernel
  int blocks = (num_odd + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  printf("Launching %d blocks × %d threads = %d total threads\n", blocks,
         THREADS_PER_BLOCK, blocks * THREADS_PER_BLOCK);

  lucas_lehmer_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_exponents, d_results,
                                                     num_odd);
  cudaDeviceSynchronize();

  // Copy results back
  cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);

  // Collect and print primes
  printf("\nFound Mersenne primes:\n");
  int count = 0;
  for (int i = 0; i < num_odd; ++i) {
    if (h_results[i]) {
      printf("  M%d = 2^%d - 1 is prime\n", h_exponents[i], h_exponents[i]);
      count++;
    }
  }
  printf("\nTotal found: %d\n", count);

  // Cleanup
  free(h_exponents);
  free(h_results);
  cudaFree(d_exponents);
  cudaFree(d_results);

  return 0;
}