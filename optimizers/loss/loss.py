import numpy as np


class Loss:
    def __init__(self):
        pass
    def evaluate_loss(self, position):
        """
        Position is the a numpy array where the loss should be evaluated
        :param position:
        :return:
        """
        pass
    def evaluate_gradient(self, position):
        """
        Position is the a numpy array where the gradient should be evaluated

        :return:
        """
        pass

    

class QuadraticForm(Loss):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def loss(self, x):
        return 0.5 * x.T @ self.A @ x - self.b.T @ x

    def gradient(self, x, batch=None):
        return self.A @ x - self.b