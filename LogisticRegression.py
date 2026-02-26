import numpy as np
from optimizers.loss.loss import LossObj

class LogisticRegression(LossObj):
    def __init__(self, data):
        self.data = data

    def evaluate_loss(self, x):
        pass

    def evaluate_gradient(self, x, batch=None):
        pass
