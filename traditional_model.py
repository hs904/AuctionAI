import numpy as np
import pandas as pd
import joblib
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('./Data/prepared_data.csv')

# Define features and target variable
X = df.drop(columns=["log_winner_expenses"], errors="ignore")
y = df["log_winner_expenses"].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test data
np.save("./Data/X_test.npy", X_test_scaled)
np.save("./Data/y_test.npy", y_test)

# Define linear models
models = {
    "OLS Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Compute performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R² Score": r2}

    # Save trained model
    joblib.dump(model, f'./Model/{name.replace(" ", "_")}.pkl')

# Load and evaluate trained neural network
from Model_training import ImprovedNeuralNetwork
model_nn = ImprovedNeuralNetwork(input_dim=X_test.shape[1])
model_nn = torch.load("./Model/trained_model.pth")
model_nn.eval()

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    y_pred_tensor = model_nn(X_test_tensor).detach().numpy().flatten()

# Compute performance metrics for NN
mae = mean_absolute_error(y_test, y_pred_tensor)
mse = mean_squared_error(y_test, y_pred_tensor)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_tensor)

results["Neural Network"] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R² Score": r2}

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
results_df.to_csv("./Evaluation_plots/model_comparison.csv", index=True)

# Print results
print("\nModel Performance Comparison:")
print(results_df)

# Visualization
plot_dir = "Evaluation_plots"
os.makedirs(plot_dir, exist_ok=True)

metrics = ["MAE", "MSE", "RMSE", "R² Score"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=results_df.index, y=results_df[metric])
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.title(f"Model Comparison: {metric}")
    plt.savefig(os.path.join(plot_dir, f"{metric}_comparison.png"))
    plt.show()