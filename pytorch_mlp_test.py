import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 1. Define the MLP Architecture
class StockPredictorMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StockPredictorMLP, self).__init__()
        
        # f vector -> g (hidden layer)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sig = nn.Sigmoid() # Activation function for the hidden layer
        
        # g -> h vector (output layer)
        # Output is size 1 since we are predicting a single continuous value (e.g., next month's return)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, f):
        # Forward pass: f -> g -> h
        g = self.fc1(f)
        g = self.sig(g)
        h = self.fc2(g)
        return h

# 2. Simulate Preprocessing Yuka's Features (The "f" vector)
def preprocess_features(ret_1m, ret3m, ret_6m, vol_3m, vol_6m, volume_z):
    # Combine into our f vector
    f_vector = [ret_1m, ret3m, ret_6m, vol_3m, vol_6m, volume_z]
    return torch.tensor(f_vector, dtype=torch.float32)

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


# tensor-ify X and y
df = pd.read_csv("X.csv")
X_np = df.values
X_tensor = torch.tensor(X_np, dtype=torch.float32)

df_y = pd.read_csv("y.csv")
y_np = df_y.values
y_tensor = torch.tensor(y_np, dtype=torch.float32)

print(X_tensor.shape)
print(y_tensor.shape)
print("Starting Training Loop...\n")

for epoch in range(EPOCHS):
    total_loss = 0
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass (Get 'h')
    prediction_h = model(X_tensor)
    
    # Calculate loss (Compare 'h' to 'truth')
    loss = criterion(prediction_h, y_tensor)
    
    # Backward pass (Backpropagation)
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    total_loss += loss.item()
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {total_loss/5:.6f}")

print("\nTraining Complete!")
