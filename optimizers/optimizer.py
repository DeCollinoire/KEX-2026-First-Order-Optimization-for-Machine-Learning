import numpy as np
from optimizers.loss.loss import LossObj

class Optimizer:
    def __init__(self, lossObj: LossObj, initPos = np.array([])):
        # Store parameters/position and loss
        self.lossObj = lossObj
        self.pos = np.array(initPos, dtype=float)

        self.lossHistory = []
        self.posHistory = []

    def __call__(self, nr_epochs=100):
        return self.optimize(nr_epochs)

    def optimize(self, nr_epochs = 100):
        """ Full optimization. Takes `nr_epochs` number of optimizer steps and stores the history."""
        # Reset history
        self.posHistory = [self.pos.copy()]
        self.lossHistory = [self.lossObj.evaluate_loss(self.pos)]
        
        # Step nr_epochs times
        for _ in range(nr_epochs):
            pos = self.step()
            self.posHistory.append(pos.copy())
            self.lossHistory.append(self.lossObj.evaluate_loss(pos))
        return np.array(self.posHistory), np.array(self.lossHistory)

    def step(self):
        return np.array([])
