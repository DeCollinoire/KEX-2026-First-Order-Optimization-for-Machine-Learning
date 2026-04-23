import numpy as np
from scipy.special import expit
from timeit import default_timer

def get_batches(data, batch_size=50):
    return [data[start:start + batch_size] for start in range(0, len(data), batch_size)]

class LossObj:
    def __init__(self, data=None):
        if data is not None:
            self.X, self.y = data  # assumes data is in (X, y) format (features, labels)
            self.nrSamples = self.X.shape[0]
            self.nrFeatures = self.X.shape[1]
        else:
            self.X, self.y = None, None
            self.nrSamples, self.nrFeatures = 0, 0

    def shuffleData(self):
        """ Used to step shuffle internal data if desired """
        indices = np.random.permutation(self.nrSamples)
        if self.X is not None and self.y is not None:
            self.X = self.X[indices]
            self.y = self.y[indices]

    def getCurrentBatch(self):
        return self.X_batch, self.y_batch

    def setCurrentBatch(self, X_batch, y_batch):
        self.X_batch = X_batch
        self.y_batch = y_batch

    def evaluate_loss(self, position):
        raise NotImplementedError
    
    def evaluate_gradient(self, position):
        raise NotImplementedError

class LogisticRegression(LossObj):
    def __init__(self, data):
        super().__init__(data)
        self.divideByBatchSize = True

    def evaluate_loss(self, position):
        return np.sum(np.log(1 + np.exp(-self.y * (self.X @ position)))) / len(self.y)

    def evaluate_gradient(self, position):
        X, y = self.getCurrentBatch()
        if self.divideByBatchSize:
            return -(X.T @ (y * expit(-(y * (X @ position))))) / len(y) # expit is a sigmoid function
        else: 
            return -(X.T @ (y * expit(-(y * (X @ position))))) # expit is a sigmoid function

#Test
if __name__ == "__main__":
    # Data is in format: np.array([x1, x2, x3...]), np.array([y1,y2,y3])
    pass
