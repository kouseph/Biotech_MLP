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

# required_fundamental_features = [
#     "total_cash",
#     "total_cash_log",
#     "total_cash_ge_1b",
#     "total_cash_trend_4q",
#     "lfcf",
#     "lfcf_trend_4q",
#     "lfcf_improving_4q",
#     "days_since_total_cash_report",
#     "days_since_lfcf_report",
#     "fundamentals_available",
# ]
# missing_required = [f for f in required_fundamental_features if f not in X_train.columns]
# if missing_required:
#     raise ValueError(
#         f"Missing new cashflow/cash features in X_train.csv: {missing_required}. "
#         "Run loaddata.py first to regenerate datasets."
#     )

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
        
        # Output layer: 64 -> 1 (continuous regression output)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x_1 = self.relu1(self.fc1(x))
        x_2 = self.relu2(self.fc2(x_1))
        x_3 = self.relu3(self.fc3(x_2))
        x_4 = self.fc4(x_3)  # predicted next-month absolute return
        return x_4

# --- 3. Initialize model, loss, optimizer ---
model = StockTop20MLP(input_size=X_train_tensor.shape[1])

# Regression objective for continuous target
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1000

# Surface data issues early if they exist upstream
if torch.isnan(X_train_tensor).any() or torch.isnan(y_train_tensor).any():
    raise ValueError("NaNs found in training tensors. Regenerate data in loaddata_j.py.")
if torch.isnan(X_test_tensor).any() or torch.isnan(y_test_tensor).any():
    raise ValueError("NaNs found in test tensors. Regenerate data in loaddata_j.py.")

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
with torch.no_grad():
    preds = model(X_test_tensor)
    mse = torch.mean((preds - y_test_tensor) ** 2)
    rmse = torch.sqrt(mse)
print("RMSE:", rmse.item())

# top_percent = 0.2
# with torch.no_grad():
#     logits = model(X_test_tensor)
#     preds_prob = torch.sigmoid(logits).squeeze()  # convert logits to probabilities
#     preds_label = (preds_prob >= top_percent).float()

# test_months = pd.read_csv("test_months.csv")["month"]
# test_fundamentals_available = pd.read_csv("test_fundamentals_available.csv")["fundamentals_available"]
# results = pd.DataFrame({
#     "month": test_months,
#     "true": y_test['target'],
#     "pred_prob": preds_prob.squeeze().numpy(),
#     "fundamentals_available": test_fundamentals_available
# })

# groups = results.groupby("month")

# monthly_hits = groups.apply(top20_hit_rate)
# print(monthly_hits)






# # --- 2. Determine K ---  # top 20%
# K = int(top_percent * len(preds_prob))

# # --- 3. Get indices of top K predictions ---
# topk_indices = torch.topk(preds_prob, K).indices

# # --- 4. Count hits ---
# y_true = y_test_tensor.squeeze()
# hits = y_true[topk_indices].sum().item()  # number of actual top20 stocks in top K

# hit_rate = hits / K
# print(hits, K)
# print(f"Top-{int(top_percent*100)}% hit rate: {hit_rate:.3f}")

# results_fund = results[results["fundamentals_available"] == 1]
# if len(results_fund) > 0:
#     monthly_hits_fund = results_fund.groupby("month").apply(top20_hit_rate)
#     fund_hits = monthly_hits_fund["hits"].sum()
#     fund_k = monthly_hits_fund["top_k"].sum()
#     fund_hit_rate = fund_hits / fund_k if fund_k > 0 else np.nan
#     print(
#         f"Fundamentals-available subset hit rate "
#         f"(monthly top-{int(top_percent*100)}% picks): {fund_hit_rate:.3f} "
#         f"[hits={fund_hits:.0f}, picks={fund_k:.0f}]"
#     )
# else:
#     print("No test rows have both total_cash and lfcf available within freshness window.")


import matplotlib.pyplot as plt

# --- 6. Regression visualizations: actual volatility vs predicted ---
y_true_np = y_test_tensor.squeeze().numpy()
y_pred_np = preds.squeeze().numpy()

# View 1: ordered sample window (easy to inspect local fit)
visualize_n = min(250, len(y_true_np))
plt.figure(figsize=(12, 4))
plt.plot(y_true_np[:visualize_n], label="Actual volatility", linewidth=2)
plt.plot(y_pred_np[:visualize_n], label="Predicted volatility", linewidth=2, alpha=0.85)
plt.title(f"Actual vs Predicted Volatility (First {visualize_n} Test Samples)")
plt.xlabel("Test sample index")
plt.ylabel("Absolute next-month return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# # View 2: scatter with perfect-fit reference line plt.figure(figsize=(6, 6))
# plt.scatter(y_true_np, y_pred_np, alpha=0.5, s=20)
# min_v = float(min(y_true_np.min(), y_pred_np.min()))
# max_v = float(max(y_true_np.max(), y_pred_np.max()))
# plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Perfect fit (y=x)")
# plt.title("Predicted vs Actual Volatility")
# plt.xlabel("Actual volatility")
# plt.ylabel("Predicted volatility")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
