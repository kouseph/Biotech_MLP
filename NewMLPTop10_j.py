import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

# --- 2. Define MLP regressor ---
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

# Surface data issues early if they exist upstream
if torch.isnan(X_train_tensor).any() or torch.isnan(y_train_tensor).any():
    raise ValueError("NaNs found in training tensors. Regenerate data in loaddata_j.py.")
if torch.isnan(X_test_tensor).any() or torch.isnan(y_test_tensor).any():
    raise ValueError("NaNs found in test tensors. Regenerate data in loaddata_j.py.")

torch.manual_seed(42)
np.random.seed(42)

def fit_and_predict(X_train_df, X_test_df, y_train_tensor_local, epochs=1000, lr=0.001):
    X_train_local = torch.tensor(X_train_df.values.astype(np.float32))
    X_test_local = torch.tensor(X_test_df.values.astype(np.float32))
    model_local = StockTop20MLP(input_size=X_train_local.shape[1])
    criterion_local = nn.MSELoss()
    optimizer_local = optim.Adam(model_local.parameters(), lr=lr)

    for epoch in range(epochs):
        model_local.train()
        optimizer_local.zero_grad()
        pred_train = model_local(X_train_local)
        loss = criterion_local(pred_train, y_train_tensor_local)
        loss.backward()
        optimizer_local.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    model_local.eval()
    with torch.no_grad():
        pred_test = model_local(X_test_local).squeeze().numpy()
    return pred_test

def regression_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residuals ** 2) / denom) if denom > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}

# --- 3. Ablation feature sets ---
quarterly_feature_markers = [
    "total_cash",
    "lfcf",
    "fundamental",
    "days_since_total_cash_report",
    "days_since_lfcf_report",
]
monthly_cols = [
    c for c in X_train.columns
    if not any(marker in c for marker in quarterly_feature_markers)
]
quarterly_level_cols = [
    c for c in X_train.columns
    if any(base in c for base in ["total_cash", "lfcf", "fundamental", "days_since_total_cash_report", "days_since_lfcf_report"])
    and all(dynamic_key not in c for dynamic_key in ["qoq", "yoy", "trend", "improving", "to_total_cash"])
]
quarterly_dynamic_cols = [
    c for c in X_train.columns
    if any(dynamic_key in c for dynamic_key in ["qoq", "yoy", "trend", "improving", "to_total_cash"])
]

feature_sets = {
    "monthly_only": monthly_cols,
    "monthly_plus_quarterly_levels": sorted(list(set(monthly_cols + quarterly_level_cols))),
    "monthly_plus_levels_plus_dynamics": sorted(
        list(set(monthly_cols + quarterly_level_cols + quarterly_dynamic_cols))
    ),
}

print("\n=== Ablation Results ===")
y_true_np = y_test_tensor.squeeze().numpy()
ablation_predictions = {}
for name, cols in feature_sets.items():
    cols = [c for c in cols if c in X_train.columns and c in X_test.columns]
    print(f"\n{name}: {len(cols)} features")
    y_pred = fit_and_predict(
        X_train_df=X_train[cols],
        X_test_df=X_test[cols],
        y_train_tensor_local=y_train_tensor,
    )
    ablation_predictions[name] = y_pred
    metrics = regression_metrics(y_true_np, y_pred)
    print(
        f"{name} -> RMSE: {metrics['rmse']:.6f} | "
        f"MAE: {metrics['mae']:.6f} | R2: {metrics['r2']:.6f}"
    )

best_name = min(
    ablation_predictions.keys(),
    key=lambda n: regression_metrics(y_true_np, ablation_predictions[n])["rmse"],
)
y_pred_np = ablation_predictions[best_name]
best_metrics = regression_metrics(y_true_np, y_pred_np)
print(
    f"\nBest model: {best_name} | RMSE: {best_metrics['rmse']:.6f} | "
    f"MAE: {best_metrics['mae']:.6f} | R2: {best_metrics['r2']:.6f}"
)
print("RMSE:", best_metrics["rmse"])

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

# --- 7. Residual diagnostics by fundamentals availability and recency ---
residuals = y_true_np - y_pred_np
fund_available = None
try:
    fund_available = pd.read_csv("test_fundamentals_available.csv")["fundamentals_available"].values
except Exception:
    fund_available = None

if fund_available is not None and len(fund_available) == len(residuals):
    available_mask = fund_available >= 0.5
    unavailable_mask = ~available_mask
    if available_mask.any():
        print(
            "Residual MAE | fundamentals_available=1:",
            float(np.mean(np.abs(residuals[available_mask]))),
        )
    if unavailable_mask.any():
        print(
            "Residual MAE | fundamentals_available=0:",
            float(np.mean(np.abs(residuals[unavailable_mask]))),
        )

recency_days = None
try:
    recency_days = pd.read_csv("test_fundamental_recency_days.csv")["fundamental_recency_days"].values
except Exception:
    recency_days = None

if recency_days is not None and len(recency_days) == len(residuals):
    recency_df = pd.DataFrame({"recency_days": recency_days, "abs_residual": np.abs(residuals)}).dropna()
    if len(recency_df) > 0:
        recency_df["recency_bin"] = pd.cut(
            recency_df["recency_days"],
            bins=[-np.inf, 30, 90, 180, np.inf],
            labels=["<=30d", "31-90d", "91-180d", ">180d"],
        )
        print("\nResidual MAE by fundamental recency:")
        print(recency_df.groupby("recency_bin", observed=False)["abs_residual"].mean())

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
