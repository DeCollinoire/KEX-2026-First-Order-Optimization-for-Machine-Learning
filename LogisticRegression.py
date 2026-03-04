import numpy as np
from optimizers.loss.loss import LossObj

class LogisticRegression(LossObj):
    def __init__(self, data):
        super().__init__(data)

    def evaluate_loss(self, position):
        return np.mean(np.log(1 + np.exp(-self.data[:, 1] * (self.data[:, 0] @ position))))

    def evaluate_loss2(self, weights):

        x = np.numarray(self.data[0])
        #Might be more stable if another log programming function is used.
        return (1/self.xDataLength) * np.sum(np.log(1 + np.exp(-self.data[1] * (x @ weights))))

    def evaluate_gradient(self, position, batch=None):
        batch = np.array(self.randomBatchList[self.currentBatch])
        return -(batch[:, 0].T @ (batch[:, 1] / (1 + np.exp(batch[:, 1] * (batch[:, 0] @ position))))) / len(batch)
        
