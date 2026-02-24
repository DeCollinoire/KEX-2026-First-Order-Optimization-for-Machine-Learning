import numpy as np
from optimizers.loss.loss import LossObj

class Optimizer:
    def __init__(self, lossObj: LossObj, initPos = np.array([])):
        # Store parameters/position and loss
        self.lossObj = lossObj
        self.pos = np.array(initPos, dtype=float)

        self.lossHistory = []
        self.posHistory = []

    def __call__(self, nr_epochs = 100):
        """ Full optimization. Takes `nr_epochs` number of optimizer steps and stores the history."""
        # Reset history
        self.lossHistory = []
        self.posHistory = []
        
        # Step nr_epochs times
        for _ in range(nr_epochs):
            pos = self.step()
            self.posHistory.append(pos.copy())
            self.lossHistory.append(self.lossObj.evaluate_loss(pos))
        return self.posHistory, self.lossHistory

    def step(self):
        return np.array([])
