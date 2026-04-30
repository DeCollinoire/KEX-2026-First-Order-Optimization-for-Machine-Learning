import numpy as np
from optimizers.loss.loss import LossObj
from scipy.special import expit

class LogisticRegression(LossObj):
    def __init__(self, data, batchSize = 1, fullbatch = False):
        super().__init__(data, batchSize, fullbatch)

    def evaluate_loss(self, position):
        # Old version: might cause rounding errors
        # z = self.y * (self.X @ position)
        # return np.mean(np.logaddexp(0, -z))
        
        # Might be more stable if another log programming function is used.
        return np.sum(np.log(1 + np.exp(-self.y * (self.X @ position)))) # / self.xDataListLength, NOTE: No divison by dataset length

    def evaluate_gradient(self, position):
        X, y = self.getCurrentBatch()
        # Alt 1
        # z = y * (X @ position)
        # return -(X.T @ (y * expit(-z))) / len(y)

        # Alt 2
        return -(X.T @ (y * expit(-(y * (X @ position))))) # / len(y) # expit is a sigmoid function. NOTE: Division by len(y) no longer needed.

        # Alt 3 (for {0, 1} classification)
        # y_proba = expit(X @ position)
        # return (y_proba - y) @ X / len(y)

if __name__=='__main__':
    from utils import setupProblem
    lgr, initPos = setupProblem('LogReg')
    lgr.evaluate_loss(initPos)
    