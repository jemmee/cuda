// nvcc nns_test.cu -o nns_test
//
// ./nns_test

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <vector>

// CUDA Kernel: Calculates L2 Distance for 10 million points
__global__ void computeDistances(const float *ref, const float *query,
                                 float *dists, int num_points, int dims) {
  // We use long long for indexing to prevent overflow at very large scales
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_points) {
    float sum = 0.0f;
    for (int i = 0; i < dims; i++) {
      float diff = ref[idx * dims + i] - query[i];
      sum += diff * diff;
    }
    dists[idx] = sqrtf(sum);
  }
}

int main() {
  const int N = 10000000; // 10,000,000 points
  const int D = 3;        // Dimensions

  std::cout << "Allocating memory for " << N << " points (~"
            << N * D * sizeof(float) / 1024 / 1024 << "MB on GPU)..."
            << std::endl;

  // 1. Host Memory (using unique_ptr or vector to handle large heap allocation
  // safely)
  std::vector<float> h_ref(N * D);
  std::vector<float> h_dists(N);
  float h_query[D] = {50.5f, 50.5f, 50.5f};

  // 2. CPU Bottleneck: Generating 30 million random floats
  std::cout << "Generating random data on CPU (this takes a moment)..."
            << std::endl;
  srand(42);
  for (long long i = 0; i < (long long)N * D; i++) {
    h_ref[i] =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0f));
  }

  // 3. Device Allocation
  float *d_ref, *d_query, *d_dists;
  cudaMalloc(&d_ref, (size_t)N * D * sizeof(float));
  cudaMalloc(&d_query, D * sizeof(float));
  cudaMalloc(&d_dists, (size_t)N * sizeof(float));

  // 4. Data Transfer (Host to Device)
  cudaMemcpy(d_ref, h_ref.data(), (size_t)N * D * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_query, h_query, D * sizeof(float), cudaMemcpyHostToDevice);

  // 5. Kernel Launch Configuration
  int blockSize = 256;
  // Use long long for grid calculation to ensure safety
  long long gridSize = (N + blockSize - 1) / blockSize;

  std::cout << "Launching Kernel on A30..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  computeDistances<<<gridSize, blockSize>>>(d_ref, d_query, d_dists, N, D);
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> gpu_duration = end - start;

  // 6. Copy Back & Find Min on CPU
  cudaMemcpy(h_dists.data(), d_dists, (size_t)N * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "Searching for minimum on CPU..." << std::endl;
  auto cpu_start = std::chrono::high_resolution_clock::now();

  float min_dist = FLT_MAX;
  int nearest_idx = -1;
  for (int i = 0; i < N; i++) {
    if (h_dists[i] < min_dist && h_dists[i] > 1e-5) {
      min_dist = h_dists[i];
      nearest_idx = i;
    }
  }

  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_search_duration =
      cpu_end - cpu_start;

  // 7. Output Results
  std::cout << "\n--- Scale: " << N << " Points ---" << std::endl;
  std::cout << "GPU Kernel Time:      " << gpu_duration.count() << " ms"
            << std::endl;
  std::cout << "CPU Final Search Time: " << cpu_search_duration.count() << " ms"
            << std::endl;
  std::cout << "Nearest Point Index:  " << nearest_idx << std::endl;
  std::cout << "Minimum Distance:     " << min_dist << std::endl;

  cudaFree(d_ref);
  cudaFree(d_query);
  cudaFree(d_dists);
  return 0;
}