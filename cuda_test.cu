// nvcc cuda_test.cu -o cuda_test

// ./cuda_test

#include <cuda_runtime.h>
#include <stdio.h>

// A simple CUDA error checking macro (optional but highly recommended)
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// --- CUDA Kernel (Code that runs on the GPU) ---
// This kernel simply adds 1.0 to every element in the array d_out
__global__ void simpleAddKernel(float *d_out, int size) {
  // Calculate a unique global thread ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    d_out[i] = d_out[i] + 1.0f;
  }
}

// --- Host Function (Code that runs on the CPU) ---
int main() {
  int nDevices = 0;

  // 1. Check for CUDA Device Availability
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));

  printf("--- 1. Basic Status Check ---\n");
  if (nDevices == 0) {
    printf("❌ No CUDA-capable devices found.\n");
    return 1;
  }
  printf("✅ CUDA is available. Total CUDA Devices Found: %d\n", nDevices);

  // Get properties of the first device (Device 0)
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("Device Name (Device 0): %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("--- --------------------- ---\n");

  // 2. Setup Data and Memory
  const int SIZE = 10;
  size_t size_bytes = SIZE * sizeof(float);
  float h_data[SIZE] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  float h_result[SIZE];

  // Allocate device memory (d_out)
  float *d_out = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_out, size_bytes));

  // Copy input data from Host to Device (H2D)
  CUDA_CHECK(cudaMemcpy(d_out, h_data, size_bytes, cudaMemcpyHostToDevice));

  // 3. Configure and Launch Kernel
  printf("--- 2. Simple Computation Test ---\n");
  int threadsPerBlock = 256;
  int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;

  printf("Launching Kernel (Blocks: %d, Threads/Block: %d)...\n", blocksPerGrid,
         threadsPerBlock);

  // Execute the kernel on the GPU
  simpleAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, SIZE);

  // Wait for the GPU to finish the computation and check for any errors during
  // kernel execution
  CUDA_CHECK(cudaDeviceSynchronize());

  // 4. Retrieve Results and Cleanup
  // Copy result data from Device to Host (D2H)
  CUDA_CHECK(cudaMemcpy(h_result, d_out, size_bytes, cudaMemcpyDeviceToHost));

  printf("Input Data: {1.0, 2.0, ..., 10.0}\n");
  printf("Result Data (Input + 1.0): {");
  for (int i = 0; i < SIZE; i++) {
    printf("%.1f%s", h_result[i], (i == SIZE - 1) ? "" : ", ");
  }
  printf("}\n");

  // Free device memory
  cudaFree(d_out);

  printf("\n✅ CUDA computation and verification successful.\n");
  return 0;
}
