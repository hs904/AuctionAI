import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Define existing plot directory
plot_dir = "Evaluation_plots"
os.makedirs(plot_dir, exist_ok=True)

# Load test data
X_test = np.load("./Data/X_test.npy")
y_test = np.load("./Data/y_test.npy")

# Convert to PyTorch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Load trained model
from Model_training import ImprovedNeuralNetwork
model = ImprovedNeuralNetwork(input_dim=X_test.shape[1])
model = torch.load("./Model/trained_model.pth")
model.eval()

# Make predictions
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor).detach().numpy().flatten()

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred_tensor)
mse = mean_squared_error(y_test, y_pred_tensor)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_tensor)

# Print performance metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot Residual Distribution
plt.figure(figsize=(6, 5))
residuals = y_test - y_pred_tensor
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.savefig(os.path.join(plot_dir, "residual_distribution.png"))
plt.show()

# Plot Predicted vs Actual
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred_tensor, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Predicted vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.savefig(os.path.join(plot_dir, "predicted_vs_actual.png"))
plt.show()

# Load training loss history
loss_history = np.load("./Model/loss_history.npy")

# Plot Training Loss
plt.figure(figsize=(6, 5))
plt.plot(loss_history, label="Training Loss", color='blue')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(plot_dir, "training_loss.png"))
plt.show()