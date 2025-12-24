import torch
import time

# 1. Choose Device (Auto-detect Mac GPU or NVIDIA GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Running Monte Carlo on: {device}")

def estimate_pi(num_samples):
    start = time.time()

    # 2. Generate random x and y coordinates between 0 and 1
    # We do this entirely on the GPU
    x = torch.rand(num_samples, device=device)
    y = torch.rand(num_samples, device=device)

    # 3. Calculate distance from origin (x^2 + y^2)
    # If dist <= 1, the point is inside the circle
    dist_squared = x**2 + y**2
    inside_circle = torch.sum(dist_squared <= 1.0)

    # 4. Final Calculation
    pi_estimate = 4.0 * inside_circle.item() / num_samples
    
    end = time.time()
    return pi_estimate, end - start

# Run simulation with 1 billion points
samples = 1_000_000_000
result, duration = estimate_pi(samples)

print(f"Estimated Pi: {result}")
print(f"Time taken: {duration:.4f} seconds")