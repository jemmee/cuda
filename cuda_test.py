import torch

def check_cuda_status():
    print("--- 1. Basic Status Check ---")
    
    # Check if a CUDA-enabled device is visible to PyTorch
    if torch.cuda.is_available():
        print("✅ CUDA is available and recognized by PyTorch.")
    else:
        print("❌ CUDA is NOT available. Check drivers/PyTorch installation.")
        return

    # Check device count and name
    device_count = torch.cuda.device_count()
    print(f"Total CUDA Devices Found: {device_count}")
    
    # Check the name of the first device (usually index 0)
    device_name = torch.cuda.get_device_name(0)
    print(f"Device Name (Device 0): {device_name}")

    # The A30x is Ampere-based, so this should show a high Compute Capability
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute Capability (A30x is Ampere): {major}.{minor}")

    # Set the device to the GPU
    device = torch.device("cuda")

    print("\n--- 2. Simple Computation Test ---")
    
    # Create two tensors (matrices) on the GPU
    a = torch.rand(5, 5, device=device)
    b = torch.rand(5, 5, device=device)
    
    # Perform a matrix multiplication on the GPU
    c = torch.matmul(a, b)
    
    # Verify the result is still on the GPU
    print(f"Tensor A Device: {a.device}")
    print(f"Result Tensor C Device: {c.device}")
    print(f"Result C (on GPU, 5x5 Matrix):\n{c}")
    
    # Check memory usage (optional but confirms usage)
    allocated = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
    cached = round(torch.cuda.memory_reserved(0) / 1024**3, 2)
    print(f"\nGPU Memory Allocated: {allocated} GB")
    print(f"GPU Memory Cached: {cached} GB")

    print("\n✅ CUDA computation successful.")


if __name__ == "__main__":
    check_cuda_status()
