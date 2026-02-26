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
