import numpy as np
from optimizers.optimizer import Optimizer
from optimizers.loss.loss import LossObj

class Nesterov(Optimizer):
    def __init__(self, lossObj: LossObj, initPos, lr = 0.1, decayFactor = 0.1):
        super().__init__(lossObj, initPos)

        # Hyperparameters
        self.lr = lr
        self.decayFactor = decayFactor
        
        # Velocity vector
        self.velocity = np.zeros_like(initPos)

    def step(self):
        look_ahead_pos = self.pos - self.decayFactor * self.velocity
        self.velocity = self.decayFactor * self.velocity + self.lr * self.lossObj.evaluate_gradient(look_ahead_pos)    # estimate the gradient at the next position
        self.pos -= self.velocity
        return self.pos

