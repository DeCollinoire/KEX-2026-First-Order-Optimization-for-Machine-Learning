import numpy as np
from optimizers.loss.loss import LossObj

class LogisticRegression(LossObj):
    def __init__(self, data, batchSize = 1):
        super().__init__(data, batchSize)

    def evaluate_loss(self, weights):
        # Old version: might cause rounding errors
        # z = self.y * (self.X @ position)
        # return np.mean(np.logaddexp(0, -z))
        
        # Might be more stable if another log programming function is used.
        x = np.array(self.data[0])
        return (1/self.xDataLength) * np.sum(np.log(1 + np.exp(-self.data[1] * (x @ weights))))

    def evaluate_gradient(self, weights):
        X, y = self.getCurrentBatch()
        return -(X.T @ (y / (1 + np.exp(y * (X @ weights))))) / len(y)

