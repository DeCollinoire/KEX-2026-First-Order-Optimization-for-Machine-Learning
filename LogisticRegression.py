import numpy as np
from optimizers.loss.loss import LossObj

class LogisticRegression(LossObj):
    def __init__(self, data):
        self.data = data

    def evaluate_loss(self, x):
        return np.mean(np.log(1 + np.exp(-self.data[:, 1] * (self.data[:, 0] @ x))))

    def evaluate_gradient(self, x, batch=None):
        return -(self.data[:, 0].T @ (self.data[:, 1] / (1 + np.exp(self.data[:, 1] * (self.data[:, 0] @ x))))) / len(self.data)
    
