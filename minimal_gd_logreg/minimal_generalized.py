import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from scipy.special import expit

""" 
NOTE: Need to add state for momentum and adam optimizers, and update the train function to handle that state. 
"""

# Logistic regression
def log_reg_loss(X, y, w):
    return np.sum(np.logaddexp(0, -y * (X @ w))) # alt. use np.mean

def log_reg_gradient(X, y, w):
    activation = expit(-y * (X @ w))
    dw = -y * activation
    return (X.T @ dw) # / X.shape[0] # NOTE: If dividing by X.shape[0], we get a really small gradient and don't get Amirreza's results

# Update rules
def sgd(w, X, y, gradient, learning_rate):
    return learning_rate * gradient

def momentum(w, X, y, learning_rate, decay_factor, velocity):
    grad = log_reg_gradient(X, y, w)
    velocity = decay_factor * velocity + learning_rate * grad
    return velocity

def nesterov(w, X, y, learning_rate, decay_factor, velocity):
    lookahead_w = w - decay_factor * velocity
    grad = log_reg_gradient(X, y, lookahead_w)
    velocity = decay_factor * velocity + learning_rate * grad
    return velocity

def adam(w, X, y, learning_rate, beta1, beta2, epsilon, m, v, t):
    grad = log_reg_gradient(X, y, w)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return learning_rate * m_hat / (np.sqrt(v_hat) + epsilon), m, v


def train(loss_fn, update_fn, X, y, epochs=100, **hyperparams):
    w = np.zeros(X.shape[1])
    history = []
    
    for i in range(epochs):
        loss = loss_fn(X, y, w)
        history.append(loss)
        
        # Update weights
        w -= update_fn(w, X, y, **hyperparams)
        
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
    loss_history = train(log_reg_loss, log_reg_gradient, X, y, lr=lr, epochs=50)
    plt.plot(loss_history, marker="o", label=f"lr = {lr}")

plt.title(f"Logistic Regression Loss, dataset = {dataset_filepath}, normalization = {normalization_on}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

