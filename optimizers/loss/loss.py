import numpy as np

def get_batches(data, batch_size=50):
    return [data[start:start + batch_size] for start in range(0, len(data), batch_size)]

class LossObj:
    def __init__(self, data = None):
        self.data = data
    def evaluate_loss(self, position):
        """
        Position is the a numpy array where the loss should be evaluated
        """
        return np.array([])
    def evaluate_gradient(self, position):
        """
        Position is the a numpy array where the gradient should be evaluated
        """
        return np.array([])

class QuadraticForm(LossObj):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def evaluate_loss(self, x):
        return 0.5 * x.T @ self.A @ x - self.b.T @ x

    def evaluate_gradient(self, x, batch=None):
        return self.A @ x - self.b
    

class Rosenbrock(LossObj):
    def __init__(self):
        pass

    def evaluate_loss(self, x):
        pass

    def evaluate_gradient(self, x, batch=None):
        pass


class LogisticRegression(LossObj):
    def __init__(self, data):
        self.data = data

    def evaluate_loss(self, x):
        pass

    def evaluate_gradient(self, x, batch=None):
        pass
