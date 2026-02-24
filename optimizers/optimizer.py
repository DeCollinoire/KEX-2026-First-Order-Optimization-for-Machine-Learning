
import numpy as np


def get_batches(data, batch_size=50):
    return [data[start:start + batch_size] for start in range(0, len(data), batch_size)]

class Optimizer:
    def __init__(self, initParams, **kwargs):
        """
        Sets up:
            Storage of parameters and loss
            Storage of hyperparameters
            Storage of data
        """
        # Store parameters and loss
        self.params = np.array(initParams, dtype=float) # Current parameters
        self.lossHistory = []
        self.paramsHistory = []

        # Hyperparameters
        self.lr = kwargs["lr"] if kwargs["lr"] else 0.1
        self.gamma = kwargs["gamma"] if kwargs["gamma"] else 0.1

        # Data config
        self.batch_size = kwargs["batch_size"] if kwargs["batch_size"] else 1
        self.data = kwargs["data"] if kwargs["data"] else []
    
    def __call__(self, lossFunc, nr_epochs):
        """ Full optimization. Takes `nr_epochs` number of optimizer steps and stores the history."""
        # Reset history
        self.lossHistory = []
        self.paramsHistory = []
        # Step nr_epochs times
        for _ in range(nr_epochs):
            params = self.step(lossFunc)
            self.paramsHistory.append(params)
            self.lossHistory.append(lossFunc.loss(params))
        return self.paramsHistory, self.lossHistory

    def step(self, LossObj):
        pass

class SGD(Optimizer):
    def __init__(self, initParams, **kwargs):
        super().__init__(initParams, **kwargs)
    def step(self, lossFunc):
        np.random.shuffle(self.data)

        # Do one adjustment for each batch
        for batch in get_batches(self.data, self.batch_size):
            grad = lossFunc.gradient(self.params, batch) # Gradient uses whole batch for averaging
            self.params -= self.lr * grad
        return self.params

class Nesterov(Optimizer):
    def __init__(self, initParams, **kwargs):
        super().__init__(initParams, **kwargs)
    def step(self, lossFunc):
        np.random.shuffle(self.data)

        # Do one adjustment for each batch
        for batch in get_batches(self.data, self.batch_size):
            grad_future = lossFunc.gradient(self.params, batch)
            grad = lossFunc.gradient(self.params - self.lr * grad_future, batch) # Gradient uses whole batch for averaging
            self.params -= self.lr * grad
        return self.params

