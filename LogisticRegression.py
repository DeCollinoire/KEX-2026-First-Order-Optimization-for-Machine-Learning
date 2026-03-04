import numpy as np
from optimizers.loss.loss import LossObj

class LogisticRegression(LossObj):
    def __init__(self, data):
        self.data = data    # Change to batches

    def evaluate_loss(self, position):
        return np.mean(np.log(1 + np.exp(-self.data[:, 1] * (self.data[:, 0] @ position))))

    def evaluate_gradient(self, position, batch=None):
        return -(self.data[:, 0].T @ (self.data[:, 1] / (1 + np.exp(self.data[:, 1] * (self.data[:, 0] @ position))))) / len(self.data)
    
