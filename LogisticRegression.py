import numpy as np
from optimizers.loss.loss import LossObj

class LogisticRegression(LossObj):
    def __init__(self, data):
        super().__init__(data)

    def evaluate_loss(self, position):
        z = self.y * (self.X @ position)
        return np.mean(np.logaddexp(0, -z))

    def evaluate_gradient(self, position):
        X, y = self.getCurrentBatch()
        z = y * (X @ position)       
        weights = -y / (1 + np.exp(z))
        return (X.T @ weights) / len(y)