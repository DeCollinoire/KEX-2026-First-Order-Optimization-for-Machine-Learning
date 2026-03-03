import numpy as np
from optimizers.loss.loss import LossObj

class LogisticRegression(LossObj):
    def __init__(self, data):
        self.data = data

    #We remember that the incomming data (self.data)is on the form [x/input vectors, y/labels]. self.data[0] has undergone numpy conversion.
    def evaluate_loss(self, weights):

        #Might be more stable if another log programming function is used.
        return (1/self.xDataLength) * np.sum(np.log(1 - np.exp(-self.data[1] * (self.data[0] @ weights))))

    def evaluate_gradient(self, weights, batch=None):


        pass
