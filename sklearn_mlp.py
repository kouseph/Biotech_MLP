import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- 1. PREP YOUR DATA ---
# Assume 'df' is your dataframe with a 'Year' column
# Features (X): PE Ratio, Market Cap, Rev Growth, Debt to Equity
# Target (y): Next Month Return
features = ['PE_Ratio', 'Market_Cap', 'Rev_Growth', 'Debt_to_Equity']
target = 'Next_Month_Return'

# Split by time as your friend suggested
train_df = df[df['Year'] <= 2020]
test_df = df[df['Year'] > 2020]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# --- 2. SCALE YOUR DATA (CRITICAL FOR MLPs) ---
# MLPs hate raw stock data because Market Cap is huge and PE is small.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. THE MLP MODEL ---
# hidden_layer_sizes=(100, 50) means two hidden layers
mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# --- 4. TRAINING (The "Under the Hood" part) ---
# This is where the model looks at X_train and compares it to y_train
mlp_reg.fit(X_train_scaled, y_train)

# --- 5. PREDICTION ---
# Now we give it data it has NEVER seen (2021-2022)
predictions = mlp_reg.predict(X_test_scaled)

print(f"First 5 predictions: {predictions[:5]}")
