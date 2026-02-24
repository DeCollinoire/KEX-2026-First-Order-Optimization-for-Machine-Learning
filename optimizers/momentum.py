import optimizer as opt
import loss.loss as lo
import numpy as np

class Momentum(opt.Optimizer):
        def __init__(self, lossObject: lo.Loss, decayFactor, learningRate):
            super().__init__()

            self.lossObj = lossObject
            self.decayFactor = decayFactor
            self.learningRate = learningRate

            #Previous moment/velocity term
            self.prevMoment = np.array([])
            #Current one
            self.moment = np.array([])

            #Placeholder
            self.gradLoss = np.array([])
            #Placeholder for position
            self.pos = np.array([])

#Fix memory
        def step(self):

            #Formula for SGD Momentum
            self.moment = self.prevMoment + self.learningRate * self.lossObj.evaluate_gradient(self.pos)
            self.pos = self.pos - self.moment

            #Update moment history
            self.prevMoment = self.moment

            return self.pos





