import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# --- Configuration ---
# 1. Device selection (Crucial for CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. Hyperparameters
INPUT_SIZE = 10
HIDDEN_SIZE = 20
OUTPUT_SIZE = 1
NUM_SAMPLES = 100000
BATCH_SIZE = 1024
LEARNING_RATE = 0.01
EPOCHS = 5

# ----------------------------------------------------------------------
# 1. Define the Simple Neural Network Model
# ----------------------------------------------------------------------
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        # Layer 1: Input to Hidden, followed by ReLU activation
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Layer 2: Hidden to Output
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ----------------------------------------------------------------------
# 2. Prepare Synthetic Data
# ----------------------------------------------------------------------
print("Preparing synthetic data...")
# Create random input features (X) and corresponding labels (Y)
X = torch.randn(NUM_SAMPLES, INPUT_SIZE)
# Generate a simple linear relationship for labels (Y = 2*X_0 + 3*X_1 + noise)
Y = (2 * X[:, 0] + 3 * X[:, 1] + 0.1 * torch.randn(NUM_SAMPLES)).unsqueeze(1)

# Convert to PyTorch Dataset and DataLoader for batch processing
dataset = torch.utils.data.TensorDataset(X, Y)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Data ready: {len(train_loader)} batches of size {BATCH_SIZE}.")

# ----------------------------------------------------------------------
# 3. Setup Model, Loss, and Optimizer
# ----------------------------------------------------------------------
model = SimpleModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
# CRITICAL: Move the model's weights to the CUDA device
model.to(DEVICE) 

# Mean Squared Error (MSE) is common for regression tasks
criterion = nn.MSELoss() 
# Adam optimizer is a standard choice
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------------------------------------------------
# 4. Training Loop and Benchmarking
# ----------------------------------------------------------------------
print("\nStarting Training...")
start_time = time.time()

for epoch in range(EPOCHS):
    total_loss = 0.0
    
    # Iterate over batches from the DataLoader
    for i, (inputs, labels) in enumerate(train_loader):
        
        # CRITICAL: Move the batch data to the CUDA device
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Zero the gradients (clear previous backward pass)
        optimizer.zero_grad()
        
        # Forward pass: Compute predicted outputs
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Update weights: Perform a single optimization step (adjusting weights)
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

end_time = time.time()
training_time_seconds = end_time - start_time
print(f"\nTraining complete in {training_time_seconds:.2f} seconds.")

# ----------------------------------------------------------------------
# 5. Verification (Test/Inference)
# ----------------------------------------------------------------------
# Generate a small set of test data
X_test = torch.randn(10, INPUT_SIZE).to(DEVICE)

# Switch model to evaluation mode (turns off dropout, batch norm, etc., if they were used)
model.eval()

with torch.no_grad():
    predictions = model(X_test)

print("\n--- Test (Inference) ---")
print("First 5 Test Inputs (first 2 features):")
print(X_test[:5, :2].cpu().numpy())
print("\nFirst 5 Model Predictions:")
print(predictions[:5].cpu().numpy())

# The loss should be low (e.g., < 0.05) indicating the model successfully learned the Y = 2*X_0 + 3*X_1 pattern.