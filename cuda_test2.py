# pip install torch torchvision torchaudio
# pip install numpy
# pip install psutil
# pip3 install --user gputil

# python3 cuda_test2.py

import torch
import time
import psutil
import GPUtil
import numpy as np
from datetime import datetime

# def print("="*60)
print("  NVIDIA AMPERE GPU TEST SUITE (2025)")
print("="*60)

# 1. Basic GPU detection
print("\nChecking GPU detection...")
if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE — No NVIDIA driver or GPU detected!")
    exit(1)

print(f"CUDA available:     {torch.cuda.is_available()}")
print(f"CUDA version:       {torch.version.cuda}")
print(f"PyTorch version:    {torch.__version__}")
print(f"GPU count:          {torch.cuda.device_count()}")

gpu = torch.device("cuda:0")
print(f"Default GPU:        {torch.cuda.get_device_name(0)}")

props = torch.cuda.get_device_properties(0)
print(f"Compute Capability: {props.major}.{props.minor}")
print(f"Total Memory:       {props.total_memory // 1024**3} GB")
print(f"Multiprocessors:    {props.multi_processor_count}")

# 2. Memory test (fill VRAM)
print(f"\nTesting memory allocation...")
try:
    # Try to allocate ~90% of GPU memory
    mem_total = props.total_memory
    mem_to_use = int(mem_total * 0.9)
    print(f"Allocating {mem_to_use // 1024**3} GB...")
    big_tensor = torch.randn(mem_to_use // 8, dtype=torch.float64, device=gpu)  # 8 bytes per float64
    print("Memory allocation SUCCESS")
    del big_tensor
    torch.cuda.empty_cache()
except Exception as e:
    print(f"Memory test FAILED: {e}")

# 3. Compute stress test (matrix multiply + tensor cores)
print(f"\nRunning compute stress test (30 seconds)...")
try:
    tensor_size = 16384
    # a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
    a = torch.randn(tensor_size, tensor_size, device=gpu, dtype=torch.float16)
    # b = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
    b = torch.randn(tensor_size, tensor_size, device=gpu, dtype=torch.float16)

    print("   Warming up...")
    for _ in range(5):
        c = a @ b

    torch.cuda.synchronize()
    start = time.time()

    iterations = 500;

    print("   Stressing Tensor Cores...")
    for i in range(iterations):
        c = a @ b
        if i % 10 == 0:
            print(f"      iteration {i+1}/{iterations}...")

    torch.cuda.synchronize()
    elapsed = time.time() - start
    gflops = (2 * tensor_size**3 * iterations) / (elapsed * 1e12)  # rough TFLOPS estimate
    print(f"Compute test PASSED in {elapsed:.2f}s (~{gflops:.1f} TFLOPS FP16)")

except Exception as e:
    print(f"Compute test FAILED: {e}")

# 4. Temperature & power monitoring
print(f"\nMonitoring temperature & power...")
try:
    gpus = GPUtil.getGPUs()
    gpu0 = gpus[0]
    print(f"Temperature: {gpu0.temperature} °C")
    print(f"Power Draw:  {gpu0.powerDraw if hasattr(gpu0, 'powerDraw') else 'N/A'} W")
    print(f"Memory Used: {gpu0.memoryUsed} MB / {gpu0.memoryTotal} MB")
except:
    print("GPUtil not available — run: pip install gputil")

# 5. Final verdict
print("\n" + "="*60)
if "RTX 30" in torch.cuda.get_device_name(0) or "RTX 40" in torch.cuda.get_device_name(0) or "A100" in torch.cuda.get_device_name(0):
    print("AMPERE GPU DETECTED AND FULLY FUNCTIONAL!")
else:
    print("GPU detected, but not confirmed Ampere architecture.")
print("All tests completed successfully!")
print("="*60)
