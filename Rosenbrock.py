import numpy as np
from optimizers.loss.loss import LossObj

class Rosenbrock(LossObj):
    def __init__(self, dim):
        """ Note that dim is only used for determining the analytical minima. The gradient and loss functions can handle 'positions' of arbitrary dimensions"""
        super().__init__()
        self.dim = dim

    def minima(self):
        return np.ones(self.dim)

    def evaluate_loss(self, position):
        # See: https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize
        return np.sum(100.0 * (position[1:] - position[:-1]**2.0)**2.0 + (1 - position[:-1])**2.0)

    def evaluate_gradient(self, position):
        # See: https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize
        xm = position[1:-1]
        xm_m1 = position[:-2]
        xm_p1 = position[2:]
        grad = np.zeros_like(position)
        grad[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        grad[0] = -400*position[0]*(position[1]-position[0]**2) - 2*(1-position[0])
        grad[-1] = 200*(position[-1]-position[-2]**2)
        return np.clip(grad, -1e3, 1e3) # Clip gradients to avoid explosions
