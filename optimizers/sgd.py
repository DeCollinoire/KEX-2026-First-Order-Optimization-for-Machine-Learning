from optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, initParams, lr):
        super().__init__(initParams)
        self.lr = lr
    def step(self, lossObj):
        # Do one adjustment for each batch
        for batch in lossObj.batches:   # FIX
            grad = lossObj.gradient(self.params, batch) # Gradient uses whole batch for averaging
            self.params -= self.lr * grad
        return self.params
