import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- 1. LOAD AND CLEAN THE DATA ---
X = pd.read_csv('X_test_scaled.csv')
y = pd.read_csv('y_test.csv')

data = pd.concat([X, y], axis=1)

data_clean = data.dropna()

# Separate them back out
features = ['ret_1m', 'ret_3m', 'ret_6m', 'vol_3m', 'vol_6m', 'volume_z']
X_clean = data_clean[features]
y_clean = data_clean['target']

split = int(len(X_clean)*.8)

X_train = X_clean.iloc[:split]
y_train = y_clean.iloc[:split]

X_test = X_clean.iloc[split:]
y_test = y_clean.iloc[split:]

print(f"Training on {len(X_train)} months, Testing on {len(X_test)} months.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_reg = MLPRegressor(
    hidden_layer_sizes=(50, 25), 
    activation='logistic', 
    solver='adam', 
    max_iter=1000,   # Increased iterations so it has time to learn
    random_state=42
)

print("Training model under the hood...")
mlp_reg.fit(X_train_scaled, y_train)

# --- 5. PREDICT AND EVALUATE ---
predictions = mlp_reg.predict(X_test_scaled)

# Let's see how close we got
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error on Test Data: {mse:.6f}")

# Compare the first 5 predictions to the actual future returns
comparison = pd.DataFrame({
    'Actual Return': y_test.values,
    'Predicted Return': predictions
})
print("\nFirst 5 Predictions vs Reality:")
print(comparison)


# --- 6. PLOT THE RESULTS ---
plt.figure(figsize=(10, 6))

# Actual True Data in Blue
plt.plot(range(len(y_test)), y_test.values, label='Actual True Data', marker='o', linestyle='-', color='blue')

# MLP Predictions in Orange
plt.plot(range(len(y_test)), predictions, label='MLP Predictions', marker='x', linestyle='--', color='orange')

plt.title('Actual vs Predicted Monthly Returns (Test Data)')
plt.xlabel('Months (in Test Period)')
plt.ylabel('Monthly Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

plt.savefig("./plots/demo_plot.png")
