import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_rcv1

# 1. Load the dataset (subset='train' for ~23k samples)
rcv1 = fetch_rcv1(subset='train')
X = rcv1.data  # Scipy CSR sparse matrix
# Use the first topic for a binary classification task
y = rcv1.target[:, 0].toarray().ravel()

# 2. Gradient Descent Parameters
n_samples, n_features = X.shape
weights = np.zeros(n_features)
learning_rate = 10.0  # High LR often needed for log-tfidf normalized data
n_iterations = 100
loss_history = []

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

# 3. Full-Batch Gradient Descent Loop
print(f"Starting GD on {n_samples} samples with {n_features} features...")

for i in range(n_iterations):
    # Linear combination (Sparse dot product)
    z = X.dot(weights)
    predictions = sigmoid(z)
    
    # Binary Cross-Entropy Loss
    loss = -np.mean(y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10))
    loss_history.append(loss)
    
    # Gradient Calculation: (1/m) * X^T * (pred - y)
    gradient = X.T.dot(predictions - y) / n_samples
    
    # Update Weights
    weights -= learning_rate * gradient
    
    if i % 20 == 0:
        print(f"Iteration {i}: Loss {loss:.4f}")

# 4. Plotting
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='royalblue', linewidth=2)
plt.title("Full-Batch Gradient Descent Convergence (RCV1)")
plt.xlabel("Iteration")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()