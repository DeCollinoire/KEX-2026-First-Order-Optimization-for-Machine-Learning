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
    def __init__(self, A = None, b = None):
        if A is None or b is None:
            self.random_qdf()
        else:
            self.A = A
            self.b = b

    def random_qdf(self, shape = (2,2)):
        # Make A positive definite
        self.A = np.abs(np.random.normal(0,1, shape))
        self.A = self.A.T @ self.A + np.eye(shape[0])
        self.b = np.random.normal(0,1, shape[0])

    def evaluate_loss(self, x):
        x = np.array(x)
        return x.T @ self.A @ x - self.b @ x

    def evaluate_gradient(self, x, batch=None):
        x = np.array(x)
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
