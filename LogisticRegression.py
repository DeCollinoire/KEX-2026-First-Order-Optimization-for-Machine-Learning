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
        return (1/self.xDataLength) * np.sum(np.log(1 + np.exp(-self.y * (self.X @ weights))))

    def evaluate_gradient(self, weights):
        X, y = self.getCurrentBatch()
        return -(X.T @ (y / (1 + np.exp(y * (X @ weights))))) / len(y)

if __name__=='__main__':
    from utils import setupProblem
    lgr, initPos = setupProblem('LogReg')
    lgr.evaluate_loss(initPos)
    