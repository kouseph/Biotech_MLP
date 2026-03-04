import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# MLP ARCHITECTURE 
class StockPredictorMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StockPredictorMLP, self).__init__()
        
        # Input -> Hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        # Hidden -> Output
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, f):
        g = self.relu(self.fc1(f))  # ReLU applied
        h = self.fc2(g)             # No activation for regression
        return h

# --- Setup and Training ---

# Hyperparameters
INPUT_FEATURES = 6 # Based on Yuka's list
HIDDEN_NODES = 32  # Size of 'g'
LEARNING_RATE = 0.001
EPOCHS = 100

# Initialize Model, Loss Function, and Optimizer
model = StockPredictorMLP(input_size=INPUT_FEATURES, hidden_size=HIDDEN_NODES)
criterion = nn.MSELoss() # Mean Squared Error to compare 'h' to the 'truth'
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# tensor-ify all csvs
X_train = pd.read_csv("X_train_scaled.csv")
X_train_np = X_train.values
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)

X_test = pd.read_csv("X_test_scaled.csv")
X_test_np = X_test.values
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

y_train = pd.read_csv("y_train.csv")
y_train_np = y_train.values
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)

y_test = pd.read_csv("y_test.csv")
y_test_np = y_test.values
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

print(X_train_tensor.shape)
print(X_test_tensor.shape)
print(y_train_tensor.shape)
print(y_test_tensor.shape)

print("Starting Training Loop...\n")

for epoch in range(EPOCHS):
    total_loss = 0
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass (Get 'h')
    prediction_h = model(X_train_tensor)
    
    # Calculate loss (Compare 'h' to 'truth')
    loss = criterion(prediction_h, y_train_tensor)
    
    # Backward pass (Backpropagation)what kind of 
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    total_loss += loss.item()
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {total_loss/5:.6f}")

print("\nTraining Complete!")


# TESTING 
model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)

print(f"\nTest MSE: {test_loss.item():.6f}")

print("test pred shape", test_predictions.shape)
print("y test shape", y_test_tensor.shape)

# Convert tensors to numpy
pred_np = test_predictions.numpy().flatten()
true_np = y_test_tensor.numpy().flatten()

# Directional accuracy
direction_pred = np.sign(pred_np)
direction_true = np.sign(true_np)

accuracy = np.mean(direction_pred == direction_true)

print(f"Directional Accuracy: {accuracy:.4f}")
