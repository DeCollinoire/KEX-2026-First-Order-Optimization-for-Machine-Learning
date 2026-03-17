import numpy as np
from optimizers.loss.loss import LossObj

class QuadraticForm(LossObj):
    def __init__(self, A = None, b = None):
        super().__init__()
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

    def minima(self):
        return np.linalg.solve(self.A, self.b)

    def evaluate_loss(self, x):
        x = np.array(x)
        return 0.5 * x.T @ self.A @ x - self.b @ x  # 0.5 factor unexpectedly made the plot look better, but it doesn't change the optimization problem.

    def evaluate_gradient(self, x, batch=None):
        x = np.array(x)
        return self.A @ x - self.b
    
