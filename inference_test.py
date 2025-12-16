import torch
import torch.nn as nn
import numpy as np
import time

# --- Configuration ---
# 1. Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. Model/Data parameters
INPUT_SIZE = 512    # Features (e.g., from a large image or sensor data)
HIDDEN_SIZE = 1024
OUTPUT_SIZE = 10    # Classes (e.g., classifying 10 types of objects)
BATCH_SIZE = 5000   # Large batch size to test GPU throughput
NUM_INFERENCE_BATCHES = 50 

# ----------------------------------------------------------------------
# 1. Define the Neural Network Model
# ----------------------------------------------------------------------
class InferenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(InferenceModel, self).__init__()
        # Simulating a moderately complex model with three layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ----------------------------------------------------------------------
# 2. Initialize and Prepare Model
# ----------------------------------------------------------------------
model = InferenceModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
# Load weights (we are just initializing random weights here for the demo)
# In a real application, you would use: model.load_state_dict(torch.load('model.pth'))

# CRITICAL: Move the model's weights to the CUDA device
model.to(DEVICE) 

# Set model to evaluation mode (crucial for accurate inference timing)
model.eval() 

print(f"Model initialized and moved to {DEVICE}. Parameters: {sum(p.numel() for p in model.parameters()):,} ")

# ----------------------------------------------------------------------
# 3. CUDA Warmup and Benchmarking
# ----------------------------------------------------------------------
# Generate a single dummy batch for warm-up
dummy_input = torch.randn(BATCH_SIZE, INPUT_SIZE).to(DEVICE)
print("\nStarting CUDA Warm-up...")

# Perform a few forward passes to warm up the CUDA cores, caches, and memory allocation.
for _ in range(5):
    with torch.no_grad(): # Disable gradient calculations
        _ = model(dummy_input)

# Synchronize CUDA to ensure all warm-up operations are truly complete
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()

# --- Start Inference Timing ---
print("Starting bulk inference test...")
start_time = time.time()

all_predictions = []

# torch.no_grad() is the most critical optimization for inference
with torch.no_grad():
    for i in range(NUM_INFERENCE_BATCHES):
        # Create a new batch of random data on the CPU
        batch_input_cpu = torch.randn(BATCH_SIZE, INPUT_SIZE)
        
        # Move batch data to the GPU
        batch_input_gpu = batch_input_cpu.to(DEVICE)
        
        # Forward pass (Inference)
        batch_output = model(batch_input_gpu)
        
        # Store predictions (Moving back to CPU is often skipped in real systems)
        all_predictions.append(batch_output.cpu())

# Final synchronization before stopping the clock
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()

end_time = time.time()
total_time = end_time - start_time

# --- Results ---
total_inferences = NUM_INFERENCE_BATCHES * BATCH_SIZE
print("\n--- Performance Results ---")
print(f"Total time for {total_inferences:,} inferences: {total_time:.4f} seconds")
print(f"Inferences per second (Throughput): {total_inferences / total_time:.2f} FPS")

# Calculate Latency (average time per sample)
latency_ms = (total_time / total_inferences) * 1000
print(f"Average Latency per sample: {latency_ms:.4f} ms")

# Example output from the last batch
last_batch_predictions = all_predictions[-1]
print("\nExample Output (First 5 predictions of last batch):")
# Use softmax to get probabilities for the classification task
probabilities = torch.softmax(last_batch_predictions[:5], dim=1)
predicted_classes = torch.argmax(probabilities, dim=1)

print("Probabilities (5 samples, 10 classes):")
print(probabilities.numpy())
print("\nPredicted Classes (0-9):")
print(predicted_classes.numpy())