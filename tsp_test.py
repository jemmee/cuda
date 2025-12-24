import torch
import numpy as np

# 1. Setup Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 2. Problem Parameters
num_cities = 20
num_paths_to_test = 100_000  # We will test 100k paths at once!

# 3. Create Cities (Random x, y coordinates)
cities = torch.rand(num_cities, 2, device=device)

def solve_tsp_parallel(cities, batch_size):
    n = cities.shape[0]
    
    # Generate batch_size number of random permutations (paths)
    # Each row is a unique path through the cities
    paths = torch.stack([torch.randperm(n, device=device) for _ in range(batch_size)])
    
    # Gather city coordinates in the order of the random paths
    # Shape becomes [batch_size, num_cities, 2]
    ordered_cities = cities[paths]
    
    # Calculate distances between consecutive cities
    # d = sqrt((x2-x1)^2 + (y2-y1)^2)
    diffs = ordered_cities[:, 1:, :] - ordered_cities[:, :-1, :]
    distances = torch.sqrt(torch.sum(diffs**2, dim=2))
    
    # Sum distances for each path
    total_distances = torch.sum(distances, dim=1)
    
    # Find the shortest one in the batch
    best_dist, best_idx = torch.min(total_distances, dim=0)
    return best_dist.item(), paths[best_idx]

# 4. Run the Solver
best_val, best_path = solve_tsp_parallel(cities, num_paths_to_test)

print(f"Shortest distance found: {best_val:.4f}")
print(f"Path taken: {best_path.tolist()}")