import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
def preprocess_features(insider_sent, analyst_rec, six_mo_return, pe_ratio, volatility, ibb_return, market_cap):
    # Scale insider sentiment from [-100, 100] to [-1, 1]
    scaled_insider = insider_sent / 100.0
    
    # Analyst rec is already -1, 0, 1 (No change needed)
    
    # Market Cap: Apply Log Scaling to turn exponential differences into linear ones
    # X_scaled = log(X)
    log_market_cap = np.log(market_cap)
    
    # (In a real scenario, you would also z-score standardise PE, Volatility, and Returns here 
    # using a scaler like sklearn's StandardScaler)
    scaler = StandardScaler()
    scaler.fit(pe_ratio)
    
    # Combine into our f vector
    f_vector = [scaled_insider, analyst_rec, six_mo_return, pe_ratio, volatility, ibb_return, log_market_cap]
    return torch.tensor(f_vector, dtype=torch.float32)

# --- Setup and Training ---

# Hyperparameters
INPUT_FEATURES = 7 # Based on Yuka's list
HIDDEN_NODES = 32  # Size of 'g'
LEARNING_RATE = 0.001
EPOCHS = 100

# Initialize Model, Loss Function, and Optimizer
model = StockPredictorMLP(input_size=INPUT_FEATURES, hidden_size=HIDDEN_NODES)
criterion = nn.MSELoss() # Mean Squared Error to compare 'h' to the 'truth'
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Dummy Data Loop (Simulating your yfinance data pipeline)
# Let's pretend we have 5 months of data for one stock
print("Starting Training Loop...\n")
for epoch in range(EPOCHS):
    total_loss = 0
    
    # Simulating a batch of 5 records
    for _ in range(5):
        # Generate some dummy data based on the feature list
        f_input = preprocess_features(
            insider_sent=45.0, 
            analyst_rec=1.0, 
            six_mo_return=0.08, 
            pe_ratio=25.5, 
            volatility=0.15, 
            ibb_return=0.02, 
            market_cap=50000000000
        )
        
        # The "Truth" (e.g., the actual normalized return for the following month from yfinance)
        truth_y = torch.tensor([0.05], dtype=torch.float32) 
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass (Get 'h')
        prediction_h = model(f_input)
        
        # Calculate loss (Compare 'h' to 'truth')
        loss = criterion(prediction_h, truth_y)
        
        # Backward pass (Backpropagation)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {total_loss/5:.6f}")

print("\nTraining Complete!")
