import torch
import pandas as pd
import matplotlib.pyplot as plt

# 1. Set the model to evaluation mode (crucial step!)
model.eval()

# Lists to store our results
predictions = []
actuals = []
dates = [] # Assuming you kept track of the timestamps

print("Evaluating model against Ground Truth...\n")

# 2. Turn off gradient calculation for testing
with torch.no_grad():
    # Assuming 'test_features' is a list of your preprocessed f vectors 
    # and 'test_targets' are the actual returns (ground truth)
    for i in range(len(test_features)):
        f_input = test_features[i]
        truth_y = test_targets[i]
        
        # Forward pass: get prediction (h vector)
        pred_h = model(f_input)
        
        # Store values (using .item() to pull the raw number out of the PyTorch tensor)
        predictions.append(pred_h.item())
        actuals.append(truth_y.item())

# 3. Create a Pandas DataFrame for easy viewing
results_df = pd.DataFrame({
    'Ground Truth': actuals,
    'MLP Prediction': predictions
})

# Calculate how far off the model was
results_df['Error'] = results_df['MLP Prediction'] - results_df['Ground Truth']

print(results_df.head(10))

# 4. Visualizing the comparison
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Ground Truth'], results_df['MLP Prediction'], alpha=0.6, color='purple')

# Plot a perfect prediction line (y = x)
max_val = max(results_df['Ground Truth'].max(), results_df['MLP Prediction'].max())
min_val = min(results_df['Ground Truth'].min(), results_df['MLP Prediction'].min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

plt.title("MLP Predictions vs. Ground Truth Stock Returns")
plt.xlabel("Actual Returns (Ground Truth)")
plt.ylabel("Predicted Returns (h vector)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
