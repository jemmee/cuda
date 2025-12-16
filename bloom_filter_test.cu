// nvcc bloom_filter_test.cu -o bloom_filter_test
//
// ./bloom_filter_test

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

// --- Configuration ---
const int NUM_HASHES = 3; // 'k' - Number of hash functions
const int BIT_VECTOR_SIZE =
    1024 * 1024; // 'm' - Total number of bits (1M bits = 128KB)
const int NUM_ITEMS_TO_ADD = 10000;
const int NUM_ITEMS_TO_CHECK = 1000;

// CUDA Error checking macro
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << ": " << #call << std::endl;                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// ----------------------------------------------------------------------
// 1. Host Utility Functions (Simple Hash Functions) - CORRECTED
// ----------------------------------------------------------------------

// Added __host__ __device__ qualifier
__host__ __device__ unsigned int fnv1a_hash(const char *data, size_t len) {
  unsigned int hash = 2166136261U; // FNV_PRIME
  for (size_t i = 0; i < len; ++i) {
    hash ^= (unsigned char)data[i];
    hash *= 16777619U; // FNV_OFFSET_BASIS
  }
  return hash;
}

// Added __host__ __device__ qualifier
__host__ __device__ unsigned int jenkins_hash(const char *data, size_t len) {
  unsigned int hash = 0;
  for (size_t i = 0; i < len; ++i) {
    hash += (unsigned char)data[i];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}

// ----------------------------------------------------------------------
// 2. CUDA Kernel: Adding Items to the Bloom Filter
// ----------------------------------------------------------------------

/**
 * @brief CUDA kernel to add multiple items to the Bloom Filter's bit vector.
 *
 * Each thread handles the hashing and bit-setting for one item (string).
 * Uses atomic operations to ensure safe concurrent bit setting.
 *
 * @param d_bit_vector_int - Device pointer to the Bloom Filter bit array (as
 * array of unsigned ints).
 * @param d_item_data - Device pointer to the concatenated string data.
 * @param d_item_lengths - Device pointer to array of item lengths.
 * @param d_item_offsets - Device pointer to array of start indices for each
 * item in d_item_data.
 * @param bit_vector_mask - Bit mask (m-1) to map hash values to valid indices.
 */
__global__ void add_to_bloom_filter_kernel(unsigned int *d_bit_vector_int,
                                           const char *d_item_data,
                                           const int *d_item_lengths,
                                           const int *d_item_offsets,
                                           unsigned int bit_vector_mask) {
  // Index of the item this thread is responsible for
  int item_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (item_idx < NUM_ITEMS_TO_ADD) {
    // --- 1. Get the item data for this thread ---
    int offset = d_item_offsets[item_idx];
    int length = d_item_lengths[item_idx];
    const char *item = d_item_data + offset;

    // --- 2. Calculate two independent base hash values (h1 and h2) ---
    // NOTE: In real-world CUDA, you'd use optimized device-side hash functions.
    // We simulate the output of two base hashes here for simplicity.
    unsigned int h1 = fnv1a_hash(item, length);
    unsigned int h2 = jenkins_hash(item, length);

    // --- 3. Compute k hash indices and set the bits ---
    for (int i = 0; i < NUM_HASHES; ++i) {
      // Diluted Hash Function: g(i) = h1 + i * h2
      unsigned int current_hash = h1 + (i * h2);

      // Map hash to the bit vector index (using mask for safety)
      unsigned int bit_index = current_hash & bit_vector_mask;

      // Calculate the word (integer) index and the bit position within the word
      unsigned int word_index = bit_index / 32;
      unsigned int bit_pos = bit_index % 32;
      unsigned int set_mask = 1U << bit_pos;

      // CRITICAL: Use atomic OR to set the bit safely.
      // This prevents race conditions where multiple threads try to write to
      // the same word.
      atomicOr(&d_bit_vector_int[word_index], set_mask);
    }
  }
}

// ----------------------------------------------------------------------
// 3. CUDA Kernel: Checking Items in the Bloom Filter
// ----------------------------------------------------------------------

__global__ void check_bloom_filter_kernel(
    unsigned int *d_bit_vector_int, const char *d_check_data,
    const int *d_check_lengths, const int *d_check_offsets,
    int *d_results, // Output: 1 if possible member, 0 otherwise
    unsigned int bit_vector_mask) {
  int item_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (item_idx < NUM_ITEMS_TO_CHECK) {
    int offset = d_check_offsets[item_idx];
    int length = d_check_lengths[item_idx];
    const char *item = d_check_data + offset;

    unsigned int h1 = fnv1a_hash(item, length);
    unsigned int h2 = jenkins_hash(item, length);

    bool is_member = true;
    for (int i = 0; i < NUM_HASHES; ++i) {
      unsigned int current_hash = h1 + (i * h2);
      unsigned int bit_index = current_hash & bit_vector_mask;

      unsigned int word_index = bit_index / 32;
      unsigned int bit_pos = bit_index % 32;
      unsigned int check_mask = 1U << bit_pos;

      // Check if the bit is set
      if ((d_bit_vector_int[word_index] & check_mask) == 0) {
        is_member = false;
        break; // Short-circuit: if one bit is not set, it's definitely not a
               // member
      }
    }

    d_results[item_idx] = is_member ? 1 : 0;
  }
}

// ----------------------------------------------------------------------
// 4. Main Host Program
// ----------------------------------------------------------------------
int main() {
  std::cout << "--- CUDA Bloom Filter Demonstration ---" << std::endl;
  std::cout << "Bit Vector Size (m): " << BIT_VECTOR_SIZE << " bits"
            << std::endl;
  std::cout << "Hash Functions (k): " << NUM_HASHES << std::endl;

  // --- Data Preparation (Host) ---
  std::vector<std::string> items_to_add;
  for (int i = 0; i < NUM_ITEMS_TO_ADD; ++i) {
    items_to_add.push_back("item_" + std::to_string(i) + "_data");
  }

  // Create non-members (for false positive testing)
  std::vector<std::string> check_items;
  for (int i = 0; i < NUM_ITEMS_TO_CHECK; ++i) {
    if (i % 2 == 0) {
      // Half are members
      check_items.push_back(items_to_add[i]);
    } else {
      // Half are guaranteed non-members
      check_items.push_back("nonmember_" + std::to_string(i));
    }
  }

  // --- Flatten Data Structures for GPU ---
  std::string all_add_data_str;
  std::vector<int> h_add_lengths(NUM_ITEMS_TO_ADD);
  std::vector<int> h_add_offsets(NUM_ITEMS_TO_ADD);
  int current_offset = 0;

  for (int i = 0; i < NUM_ITEMS_TO_ADD; ++i) {
    h_add_offsets[i] = current_offset;
    h_add_lengths[i] = items_to_add[i].length();
    all_add_data_str += items_to_add[i];
    current_offset += h_add_lengths[i];
  }

  // Do the same for check items
  std::string all_check_data_str;
  std::vector<int> h_check_lengths(NUM_ITEMS_TO_CHECK);
  std::vector<int> h_check_offsets(NUM_ITEMS_TO_CHECK);
  current_offset = 0;

  for (int i = 0; i < NUM_ITEMS_TO_CHECK; ++i) {
    h_check_offsets[i] = current_offset;
    h_check_lengths[i] = check_items[i].length();
    all_check_data_str += check_items[i];
    current_offset += h_check_lengths[i];
  }

  // Bit Vector (m bits) stored as an array of unsigned ints (m/32 words)
  const int BIT_VECTOR_WORDS = BIT_VECTOR_SIZE / 32;
  std::vector<unsigned int> h_bit_vector(BIT_VECTOR_WORDS, 0U);

  // --- Device Allocation ---
  unsigned int *d_bit_vector_int;
  char *d_add_data, *d_check_data;
  int *d_add_lengths, *d_add_offsets, *d_check_lengths, *d_check_offsets,
      *d_check_results;

  CHECK_CUDA(
      cudaMalloc(&d_bit_vector_int, BIT_VECTOR_WORDS * sizeof(unsigned int)));
  CHECK_CUDA(cudaMalloc(&d_add_data, all_add_data_str.length() * sizeof(char)));
  CHECK_CUDA(
      cudaMalloc(&d_check_data, all_check_data_str.length() * sizeof(char)));
  CHECK_CUDA(cudaMalloc(&d_add_lengths, NUM_ITEMS_TO_ADD * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_add_offsets, NUM_ITEMS_TO_ADD * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_check_lengths, NUM_ITEMS_TO_CHECK * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_check_offsets, NUM_ITEMS_TO_CHECK * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_check_results, NUM_ITEMS_TO_CHECK * sizeof(int)));

  // Initialize device bit vector to zero
  CHECK_CUDA(
      cudaMemset(d_bit_vector_int, 0, BIT_VECTOR_WORDS * sizeof(unsigned int)));

  // --- Device Transfers ---
  CHECK_CUDA(cudaMemcpy(d_add_data, all_add_data_str.c_str(),
                        all_add_data_str.length() * sizeof(char),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_check_data, all_check_data_str.c_str(),
                        all_check_data_str.length() * sizeof(char),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_add_lengths, h_add_lengths.data(),
                        NUM_ITEMS_TO_ADD * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_add_offsets, h_add_offsets.data(),
                        NUM_ITEMS_TO_ADD * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_check_lengths, h_check_lengths.data(),
                        NUM_ITEMS_TO_CHECK * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_check_offsets, h_check_offsets.data(),
                        NUM_ITEMS_TO_CHECK * sizeof(int),
                        cudaMemcpyHostToDevice));

  // --- CUDA Execution: ADD Phase ---
  int threads_per_block = 256;
  int blocks_add =
      (NUM_ITEMS_TO_ADD + threads_per_block - 1) / threads_per_block;
  unsigned int bit_vector_mask =
      BIT_VECTOR_SIZE - 1; // Mask for % BIT_VECTOR_SIZE

  std::cout << "\nStarting ADD phase (parallel execution on "
            << NUM_ITEMS_TO_ADD << " items)..." << std::endl;
  add_to_bloom_filter_kernel<<<blocks_add, threads_per_block>>>(
      d_bit_vector_int, d_add_data, d_add_lengths, d_add_offsets,
      bit_vector_mask);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::cout << "ADD phase complete." << std::endl;

  // --- CUDA Execution: CHECK Phase ---
  int blocks_check =
      (NUM_ITEMS_TO_CHECK + threads_per_block - 1) / threads_per_block;

  std::cout << "Starting CHECK phase (parallel execution on "
            << NUM_ITEMS_TO_CHECK << " items)..." << std::endl;
  check_bloom_filter_kernel<<<blocks_check, threads_per_block>>>(
      d_bit_vector_int, d_check_data, d_check_lengths, d_check_offsets,
      d_check_results, bit_vector_mask);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::cout << "CHECK phase complete." << std::endl;

  // --- Results Verification ---
  std::vector<int> h_check_results(NUM_ITEMS_TO_CHECK);
  CHECK_CUDA(cudaMemcpy(h_check_results.data(), d_check_results,
                        NUM_ITEMS_TO_CHECK * sizeof(int),
                        cudaMemcpyDeviceToHost));

  int members_found = 0;
  int false_positives = 0;

  // Check results (guaranteed members are at even indices, guaranteed
  // non-members at odd)
  for (int i = 0; i < NUM_ITEMS_TO_CHECK; ++i) {
    if (i % 2 == 0) {
      // Should be a member (Must be 1)
      if (h_check_results[i] == 1)
        members_found++;
    } else {
      // Should be a non-member (False Positive if 1)
      if (h_check_results[i] == 1)
        false_positives++;
    }
  }

  std::cout << "\n--- Verification Results ---" << std::endl;
  std::cout << "Guaranteed Members Found (True Positives): " << members_found
            << " out of " << NUM_ITEMS_TO_CHECK / 2 << std::endl;
  std::cout << "False Positives Detected: " << false_positives << " out of "
            << NUM_ITEMS_TO_CHECK / 2 << std::endl;

  // Expected False Positive Rate (p = (1 - e^(-k*n/m))^k) - for comparison
  double k = NUM_HASHES;
  double n = NUM_ITEMS_TO_ADD;
  double m = BIT_VECTOR_SIZE;
  double expected_fp_rate = std::pow(1.0 - std::exp(-k * n / m), k);
  std::cout << "Expected False Positive Rate: " << expected_fp_rate * 100 << "%"
            << std::endl;

  // --- Cleanup ---
  cudaFree(d_bit_vector_int);
  cudaFree(d_add_data);
  cudaFree(d_check_data);
  cudaFree(d_add_lengths);
  cudaFree(d_add_offsets);
  cudaFree(d_check_lengths);
  cudaFree(d_check_offsets);
  cudaFree(d_check_results);

  return 0;
}