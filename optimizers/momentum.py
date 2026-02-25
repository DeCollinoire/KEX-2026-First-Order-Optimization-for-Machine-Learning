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
            self.moment = np.array([])

            #Placeholder
            self.gradLoss = np.array([])
            #Placeholder for position
            self.pos = np.array([])

#Fix memory
        def step(self):

            #Formula for SGD Momentum
            self.moment = self.decayFactor * self.moment + self.learningRate * self.lossObj.evaluate_gradient(self.pos)
            self.pos = self.pos - self.moment


            return self.pos





