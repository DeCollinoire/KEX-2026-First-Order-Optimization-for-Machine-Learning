from optimizers.optimizer import Optimizer
from optimizers.loss.loss import LossObj

class SGD(Optimizer):
    def __init__(self, lossObj: LossObj, initPos, lr):
        super().__init__(lossObj, initPos)

        # Hyperparameters
        self.lr = lr

    def getHyperparamStr(self):
        return f"lr: {self.lr}"

    def getHyperparamDict(self):
        return {"lr": self.lr}


    def step(self):
        grad = self.lossObj.evaluate_gradient(self.pos)
        self.pos -= self.lr * grad
        return self.pos


    def reset(self):
        # Reset history & position
        self.resetHistory()
        self.resetPosition()