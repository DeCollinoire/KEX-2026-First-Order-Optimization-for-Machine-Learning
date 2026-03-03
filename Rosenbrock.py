import numpy as np
from optimizers.loss.loss import LossObj

class Rosenbrock(LossObj):

    #f(x,y)={(a-x)^2}+b{(y-{x^2})^2}}
    def __init__(self, a=1, b=100):
        self.a = a
        self.b = b


    def evaluate_loss(self, x):
        #x = pos
        return ((self.a - x[0]) ** 2) + self.b * ((x[1] - (x[1] - (x[0] ** 2))) ** 2)

    def evaluate_gradient(self, x, batch=None):
        #grad((a - x)^2 + b (y - x^2)^2) = (-2 a + 2 x + 4 b x^3 - 4 b x y, 2 b (- x^2 + y))
        return np.array([-2 * self.a + 2 * x[0] + 4 * self.b * (x[0] ** 3) - 4 * self.b * x[0] * x[1], 2 * self.b * (- (x[0] ** 2) + x[1]) ])
