// perfect_numbers_cuda.cu
// CUDA demo: Find perfect numbers up to a large limit using parallel search on
// GPU A perfect number is a positive integer equal to the sum of its proper
// divisors (excluding itself). Known perfect numbers are even and of the form
// 2^(p-1) * (2^p - 1) where (2^p - 1) is Mersenne prime. This demo brute-forces
// divisor sums in parallel to demonstrate GPU acceleration.
//
// nvcc perfect_numbers_test.cu -o perfect_numbers_test
// ./perfect_numbers_test

#include <cuda_runtime.h>
#include <stdio.h>

#define LIMIT 100000000ULL
#define THREADS_PER_BLOCK 256

// Kernel: Check if a number is perfect by summing divisors in parallel
// Each thread checks one candidate number
__global__ void find_perfect_numbers(unsigned long long start,
                                     unsigned long long end, int *results,
                                     int *count) {
  unsigned long long num = start + blockIdx.x * blockDim.x + threadIdx.x;
  if (num > end || num < 2)
    return; // Skip 0,1 and beyond limit

  unsigned long long sum = 1; // 1 is always a proper divisor

  // Check divisors up to sqrt(num)
  for (unsigned long long i = 2; i * i <= num; ++i) {
    if (num % i == 0) {
      sum += i;
      if (i != num / i && num / i != num) {
        sum += num / i;
      }
    }
  }

  if (sum == num) {
    int idx = atomicAdd(count, 1); // Thread-safe increment
    if (idx < 10) { // Store up to first 10 found (there are very few)
      results[idx] = (int)num;
    }
  }
}

int main() {
  printf("Searching for perfect numbers from 2 to %llu...\n", LIMIT);

  int *d_results, *d_count;
  int h_count = 0;
  int h_results[10] = {0};

  cudaMalloc(&d_results, 10 * sizeof(int));
  cudaMalloc(&d_count, sizeof(int));

  cudaMemset(d_count, 0, sizeof(int));
  cudaMemset(d_results, 0, 10 * sizeof(int));

  // Launch enough threads to cover the range
  unsigned long long total_numbers = LIMIT - 1;
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid((total_numbers + block.x - 1) / block.x);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  find_perfect_numbers<<<grid, block>>>(2, LIMIT, d_results, d_count);
  cudaEventRecord(stop);

  cudaDeviceSynchronize();

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_results, d_results, min(h_count, 10) * sizeof(int),
             cudaMemcpyDeviceToHost);

  printf("Search completed in %.3f ms\n", milliseconds);
  printf("Found %d perfect number(s):\n", h_count);
  for (int i = 0; i < h_count && i < 10; ++i) {
    printf("  %d\n", h_results[i]);
  }

  if (h_count == 0) {
    printf("  (None found in range – expected for small limits)\n");
  }

  // Known small perfect numbers for reference:
  // 6, 28, 496, 8128, 33550336, ...

  cudaFree(d_results);
  cudaFree(d_count);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}