// sudo apt-get update
// sudo apt-get install nvidia-gds
//
// fallocate -l 10G /data/10gb.bin
//
// nvcc gds_test.cu -o gds_test -I/usr/local/cuda-12/include
// -L/usr/local/cuda-12/lib64 -lcufile -lcudart
//
// ./gds_test

#include <chrono>
#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <iostream>
#include <string.h>
#include <unistd.h>

// Macros for clean error handling
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << std::endl;                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_CUFILE(call)                                                     \
  {                                                                            \
    CUfileError_t err = call;                                                  \
    if (err.err != CU_FILE_SUCCESS) {                                          \
      std::cerr << "cuFile Error: " << err.err << " at line " << __LINE__      \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  }

int main() {
  // 1. Setup 10GB size with 64-bit literals to prevent overflow warnings
  const size_t fileSize = 10ULL * 1024ULL * 1024ULL * 1024ULL;
  const char *filePath = "/data/10gb.bin";

  std::cout << "--- GPUDirect Storage 10GB Read Demo ---" << std::endl;
  std::cout << "Source: " << filePath << std::endl;

  // 2. Allocate GPU Memory (HBM2)
  void *d_ptr;
  CHECK_CUDA(cudaMalloc(&d_ptr, fileSize));

  // 3. Open File with O_DIRECT (Critical for GDS)
  int fd = open(filePath, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    perror("FAILED to open file at /data/10gb.bin");
    std::cerr << "Hint: Run 'sudo fallocate -l 10G /data/10gb.bin' if it "
                 "doesn't exist."
              << std::endl;
    cudaFree(d_ptr);
    return 1;
  }

  // 4. Initialize the cuFile Driver
  CHECK_CUFILE(cuFileDriverOpen());

  // 5. Register File (Explicit assignments to fix the 'int to enum' error)
  CUfileDescr_t cfr_desc;
  memset(&cfr_desc, 0, sizeof(CUfileDescr_t)); // Clear all bits

  // Assign the enum value to the type field
  cfr_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  // Assign the integer file descriptor to the union handle
  cfr_desc.handle.fd = fd;

  CUfileHandle_t cfr_handle;
  CHECK_CUFILE(cuFileHandleRegister(&cfr_handle, &cfr_desc));

  // 6. Perform the Direct DMA Transfer
  std::cout << "Transferring 10GB directly from NVMe to GPU..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // This call bypasses the CPU and System RAM entirely
  ssize_t ret = cuFileRead(cfr_handle, d_ptr, fileSize, 0, 0);

  auto end = std::chrono::high_resolution_clock::now();

  if (ret < 0) {
    std::cerr << "GDS Read Failed with error code: " << ret << std::endl;
  } else {
    std::chrono::duration<double> diff = end - start;
    double gb = (double)ret / (1024.0 * 1024.0 * 1024.0);
    std::cout << "--- Transfer Results ---" << std::endl;
    std::cout << "Total Read: " << gb << " GB" << std::endl;
    std::cout << "Total Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Bandwidth:  " << gb / diff.count() << " GB/s" << std::endl;
  }

  // 7. Cleanup
  cuFileHandleDeregister(cfr_handle);
  cuFileDriverClose();
  close(fd);
  CHECK_CUDA(cudaFree(d_ptr));

  return 0;
}