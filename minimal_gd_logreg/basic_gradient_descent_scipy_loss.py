import numpy as np
from sklearn.metrics import log_loss 

# Initialize the loss object for binary classification
# 'constant_hessian=True' makes it efficient for standard Logistic Regression
loss_func = LogLoss(sample_weight=None)

def train(X, y, lr=1.0, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    # Scikit-learn's LogLoss expects y in [0, 1] range
    # If your y is [-1, 1], convert it:
    y_proc = (y + 1) / 2
    
    history = []
    for i in range(epochs):
        # 1. Predict raw scores (X @ w)
        raw_predictions = X @ w
        
        # 2. Compute Loss
        loss = loss_func.loss(y_true=y_proc, raw_prediction=raw_predictions)
        # sklearn returns an array of individual losses; we need the mean
        avg_loss = np.mean(loss)
        history.append(avg_loss)
        
        # 3. Compute Gradient
        # returns the derivative w.r.t raw_predictions
        grad_per_sample = loss_func.gradient(y_true=y_proc, raw_prediction=raw_predictions)
        
        # Chain rule: grad_w = X.T @ grad_per_sample
        grad_w = (X.T @ grad_per_sample) / n_samples
        
        # 4. Update
        w -= lr * grad_w
        
    return history