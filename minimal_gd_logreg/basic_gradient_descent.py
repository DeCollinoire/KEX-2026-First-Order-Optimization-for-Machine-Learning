import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from scipy.special import expit

def compute_loss(X, y, w):
    return np.sum(np.logaddexp(0, -y * (X @ w))) # alt. use np.mean

def compute_gradient(X, y, w):
    activation = expit(-y * (X @ w))
    dw = -y * activation
    return (X.T @ dw) / X.shape[0] # NOTE: If dividing by X.shape[0], we get a really small gradient and don't get Amirreza's results

def update_weights(w):
    pass

def train(X, y, lr=1.0, epochs=100):
    w = np.zeros(X.shape[1])
    history = []
    
    for i in range(epochs):
        loss = compute_loss(X, y, w)
        history.append(loss)
        
        # Compute gradient and update weights (full batch)
        grad = compute_gradient(X, y, w)
        w -= lr * grad
        
        if i % 10 == 0:
            print(f"Iteration {i}: Loss {loss:.6f}")
    return history

# Config
dataset_filepath = "datasets/rcv1_train.binary" # rcv1_train.binary, australian_scale
normalization_on = True

# Load Dataset, optional normalization
X, y = load_svmlight_file(dataset_filepath) # type: ignore
if normalization_on:
    X = normalize(X, norm='l2', axis=1)

    # Alt. 2
    # scaler = MaxAbsScaler()
    # X = scaler.fit_transform(X)

# Train and plot
plt.figure()
for lr in [1e-3, 1e-2, 5e-2, 1e-1]: # amirreza's choice for australian: [1e-3, 2e-3, 8e-3, 3e-2]
    loss_history = train(X, y, lr=lr, epochs=50)
    plt.plot(loss_history, marker="o", label=f"lr = {lr}")

plt.title(f"Logistic Regression Loss, dataset = {dataset_filepath}, normalization = {normalization_on}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

