import random

import numpy as np

def get_batches(data, batch_size=50):
    return [data[start:start + batch_size] for start in range(0, len(data), batch_size)]

class LossObj:
    def __init__(self, data = None, batchAmount = 1):
        self.data = np.array(data) if data is not None else np.array([])
        self.batchAmount = batchAmount

        self.randomBatchList = []

    def minima(self):
        return None # calculate if possible

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

    def fillRandomBatchList(self):
        # shuffle data and split into batches
        np.random.shuffle(self.data)
        self.randomBatchList = get_batches(self.data, batch_size=len(self.data) // self.batchAmount)
        