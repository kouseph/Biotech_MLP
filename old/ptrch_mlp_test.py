import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# MLP ARCHITECTURE 
class StockTop20MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StockTop20MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # output probability
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# --- Setup and Training ---

# Hyperparameters
INPUT_FEATURES = 253 # Based on Yuka's list
HIDDEN_NODES = 64  # Size of 'g'
LEARNING_RATE = 0.001
EPOCHS = 100

# Initialize Model, Loss Function, and Optimizer
model = StockTop20MLP(input_size=INPUT_FEATURES, hidden_size=HIDDEN_NODES)
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# tensor-ify all csvs
X_train = pd.read_csv("X_train.csv")
X_train_np = X_train.values
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)

X_test = pd.read_csv("X_test.csv")
X_test_np = X_test.values
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

y_train = pd.read_csv("y_train.csv")
y_train_np = y_train.values
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)

y_test = pd.read_csv("y_test.csv")
y_test_np = y_test.values
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

print("X train", X_train_tensor.shape)
print("X test", X_test_tensor.shape)
print("y train", y_train_tensor.shape)
print("y test", y_test_tensor.shape)

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
    preds = model(X_test_tensor)
    preds_label = (preds >= 0.25).float()
    accuracy = (preds_label == y_test_tensor).float().mean()
print("Top-20% Classification Accuracy:", accuracy.item())


print("observed pred shape", preds.shape)
print(preds)
print("y test shape", y_test_tensor.shape)

# Convert tensors to numpy
pred_np = preds.numpy().flatten()
true_np = y_test_tensor.numpy().flatten()

# Directional accuracy
direction_pred = np.sign(pred_np)
direction_true = np.sign(true_np)

accuracy = np.mean(direction_pred == direction_true)

print(f"Directional Accuracy: {accuracy:.4f}")



n = 50
true_labels = y_test_tensor[:n].numpy().flatten()
pred_labels = preds_label[:n].numpy().flatten()

plt.figure(figsize=(12,6))
plt.plot(range(n), true_labels, marker='o', linestyle='-', color='blue', label='True Top-20%')
plt.plot(range(n), pred_labels, marker='x', linestyle='--', color='orange', label='Predicted Top-20%')
plt.title("Top-20% Classification: Predicted vs True (First 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Top-20% Label")
plt.yticks([0,1], ["No","Yes"])
plt.legend()
plt.grid(alpha=0.3)
plt.show()


