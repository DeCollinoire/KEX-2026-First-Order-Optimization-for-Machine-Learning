

class Nesterov(Optimizer):
    def __init__(self, initParams, **kwargs):
        super().__init__(initParams, **kwargs)
        self.prevVelocity = 0
    def step(self, lossFunc):
        np.random.shuffle(self.data)

        # Do one adjustment for each batch
        for batch in get_batches(self.data, self.batch_size):
            grad_future = lossFunc.gradient(self.params, batch)
            grad = lossFunc.gradient(self.params - self.lr * grad_future, batch) # Gradient uses whole batch for averaging
            self.params -= self.lr * grad
        return self.params

