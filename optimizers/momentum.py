import optimizers.optimizer as opt
import optimizers.loss.loss as lo
import numpy as np

class Momentum(opt.Optimizer):
    def __init__(self, lossObject: lo.LossObj, initPos, decayFactor, learningRate):
        super().__init__(lossObject, initPos)

        self.lossObj = lossObject
        self.decayFactor = decayFactor
        self.learningRate = learningRate

        #Moment/velocity term
        self.moment = np.zeros_like(initPos)

        #Placeholder for position
        #self.pos = initPos

    #Fix memory
    def step(self):

        #Formula for SGD Momentum
        self.moment = self.decayFactor * self.moment + self.learningRate * self.lossObj.evaluate_gradient(self.pos)
        self.pos = self.pos - self.moment

        return self.pos

    def getHyperparamStr(self):
        return f"lr: {self.learningRate}, decayfactor: {self.decayFactor}"

    def getHyperparamDict(self):
        return {"learningRate": self.learningRate, "decayFactor": self.decayFactor}

