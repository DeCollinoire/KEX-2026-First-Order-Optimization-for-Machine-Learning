
class Loss:
    def __init__(self, data = None):
        self.data = data
    
    def __call__(self, *args, **kwds):
        return self.loss(args)

    def loss(self, x):
        return 0

    def gradient(self, x):
        return 0

class QuadraticForm(Loss):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def loss(self, x):
        return 0.5 * x.T @ self.A @ x - self.b.T @ x

    def gradient(self, x, batch=None):
        return self.A @ x - self.b
    
