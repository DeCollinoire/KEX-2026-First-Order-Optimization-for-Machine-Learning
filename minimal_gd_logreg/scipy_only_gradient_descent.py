import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import numpy as np

# Load Dataset
X, y = load_svmlight_file("datasets/rcv1_train.binary") # rcv1_train.binary, australian_scale
classes = np.unique(y)

plt.figure(figsize=(9, 5))
for lr in [1e-3, 2e-3, 8e-3, 3e-2, 100, 1000]:
    # Setup SGDClassifier 
    # We disable the internal loop (max_iter=1) to control it ourselves, and penalty=None means no regularization
    clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=lr, max_iter=1, penalty=None)

    # Training Loop
    epochs = 50
    loss_history = []
    for epoch in range(epochs):
        # partial_fit runs one epoch of Stochastic Gradient Descent
        clf.partial_fit(X, y, classes=classes)
        
        # Calculate current loss across the whole dataset
        # clf.predict_proba returns [prob_class_neg, prob_class_pos]
        probs = clf.predict_proba(X)
        current_loss = log_loss(y, probs)
        loss_history.append(current_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {current_loss:.4f}")

    # 4. Plotting
    plt.plot(loss_history, marker='o', markersize=4, linestyle='-',  label=f"lr = {lr}")
    plt.title("Convergence Profile: SGDClassifier (Logistic Loss)")
    plt.xlabel("Epochs (Passes over Data)")
    plt.ylabel("Log Loss")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
plt.show()