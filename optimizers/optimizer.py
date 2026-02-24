
import numpy as np

def get_batches(data, batch_size=50):
    return [data[start:start + batch_size] for start in range(0, len(data), batch_size)]

class Optimizer:
    def __init__(self, initPos = np.array([])):
        """
        Sets up:
            Storage of parameters and loss
            Storage of hyperparameters
            Storage of data
        """
        # Store parameters and loss
        self.pos = np.array(initPos, dtype=float) # Current parameters
        self.lossHistory = []
        self.paramsHistory = []

    def __call__(self, lossFunc, nr_epochs):
        """ Full optimization. Takes `nr_epochs` number of optimizer steps and stores the history."""
        # Reset history
        self.lossHistory = []
        self.paramsHistory = []
        
        # Step nr_epochs times
        for _ in range(nr_epochs):
            params = self.step(lossFunc)
            self.paramsHistory.append(params)
            self.lossHistory.append(lossFunc.loss(params))
        return self.paramsHistory, self.lossHistory

    def step(self, LossObj):
        pass
