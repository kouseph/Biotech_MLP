import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


# --- 1. Load CSVs ---
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test  = pd.read_csv("y_test.csv")

# Convert to numpy arrays
X_train_np = X_train.values.astype(np.float32)
X_test_np  = X_test.values.astype(np.float32)
y_train_np = y_train.values.reshape(-1,1).astype(np.float32)
y_test_np  = y_test.values.reshape(-1,1).astype(np.float32)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_np)
X_test_tensor  = torch.tensor(X_test_np)
y_train_tensor = torch.tensor(y_train_np)
y_test_tensor  = torch.tensor(y_test_np)

# --- 2. Define MLP with 2 hidden layers ---
class StockTop20MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Hidden layer 1: input_size -> 256
        print(input_size)
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        
        # Hidden layer 2: 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        
        # Hidden layer 3: 128 -> 64
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        
        # Output layer: 64 -> 1 (raw logits)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)  # raw logits for BCEWithLogitsLoss
        return x

# --- 3. Initialize model, loss, optimizer ---
model = StockTop20MLP(input_size=X_train_tensor.shape[1])

# Compute pos_weight for top-20% class
pos_weight = (y_train_tensor == 0).sum() / (y_train_tensor == 1).sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1000

# --- 4. Training loop ---
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# --- 5. Evaluation ---
# model.eval()
# with torch.no_grad():
#     logits = model(X_test_tensor)
#     preds_prob = torch.sigmoid(logits)  # convert logits to probabilities
    
#     threshold = 0.25  # top-20% prevalence
#     preds_label = (preds_prob >= threshold).float()
    
#     accuracy = (preds_label == y_test_tensor).float().mean()
#     print("Top-20% Classification Accuracy:", accuracy.item())


def top20_hit_rate(group):
    n = len(group)  # total stocks that month
    k = int(n * 0.2)  # number of top picks
    if k == 0:  # handle months with very few stocks
        k = 1
    top = group.nlargest(k, "pred_prob")
    hits = top["true"].sum()  # numerator
    return pd.Series({
        "hits": hits,
        "top_k": k  # denominator
    })
def precision_at_k(group):

    k = int(len(group) * 0.2)
    
    top = group.nlargest(k, "pred_prob")
    
    return top["true"].sum() / k

model.eval()
top_percent = 0.2
with torch.no_grad():
    logits = model(X_test_tensor)
    preds_prob = torch.sigmoid(logits).squeeze()  # convert logits to probabilities
    preds_label = (preds_prob >= top_percent).float()

test_months = pd.read_csv("test_months.csv")["month"]
results = pd.DataFrame({
    "month": test_months,
    "true": y_test['target'],
    "pred_prob": preds_prob.squeeze().numpy()
})

groups = results.groupby("month")

monthly_hits = groups.apply(top20_hit_rate)
print(monthly_hits)






# --- 2. Determine K ---  # top 20%
K = int(top_percent * len(preds_prob))

# --- 3. Get indices of top K predictions ---
topk_indices = torch.topk(preds_prob, K).indices

# --- 4. Count hits ---
y_true = y_test_tensor.squeeze()
hits = y_true[topk_indices].sum().item()  # number of actual top20 stocks in top K

hit_rate = hits / K
print(hits, K)
print(f"Top-{int(top_percent*100)}% hit rate: {hit_rate:.3f}")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- 1. Predicted vs True Flags ---
visualize = 100
plt.figure(figsize=(12,4))
plt.plot(preds_label.numpy()[:visualize], 'o', label='Predicted Top-10%')
plt.plot(y_test_tensor.numpy()[:visualize], 'x', label='True Top-10%')
plt.title("Predicted vs True Top-20% Flags (First 100 Test Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Top-20% Flag (1=Yes, 0=No)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 2. Confusion Matrix ---
cm = confusion_matrix(y_test_tensor.numpy(), preds_label.numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Top-10% Classification Confusion Matrix")
plt.show()

# # --- 3. Optional: Monthly Accuracy ---
# If you have a 'month' column in X_test
if 'month' in X_test.columns:
    X_test['pred'] = preds_label.numpy()
    X_test['true'] = y_test_tensor.numpy()
    monthly_acc = X_test.groupby('month').apply(lambda df: (df['pred']==df['true']).mean())
    
    plt.figure(figsize=(12,4))
    monthly_acc.plot(marker='o')
    plt.title("Monthly Accuracy of Top-20% Prediction")
    plt.xlabel("Month")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.show()