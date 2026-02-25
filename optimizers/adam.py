import optimizers.optimizer as opt
import optimizers.loss.loss as lo
import numpy as np

class Adam(opt.Optimizer):
    def __init__(self, lossObject: lo.LossObj, learningRate, initPos, forgettingFactorM = 0.9, forgettingFactorR = 0.999):
        """
        The forgettingFactorM is for the momentum part in adam while the other forgetting factor is for the RMSprop part
        :param lossObject:
        :param forgettingFactorM:
        :param forgettingFactorR:
        :param learningRate:
        """
        super().__init__(lossObject, initPos)

        self.lossObj = lossObject
        self.decayFactor = forgettingFactorM
        self.forgettingfactorRMS = forgettingFactorR
        self.learningRate = learningRate


        # Moment/velocity term
        self.moment = np.array([])
        #For the bias corrected term
        self.biasCorrectedMoment = np.array([])

        #For the second moment in adam, for the rms part
        self.RMSMoment = np.array([])
        #For the bias corrected term
        self.biasCorrectedRMSM = np.array([])

        # Placeholder for position
        self.pos = initPos

        #For avoiding zero division in adam
        self.epsilon = 10 ** -2

    def step(self, iteration):


            #Moment term
            self.moment = self.decayFactor * self.moment + (1 - self.decayFactor) * self.lossObj.evaluate_gradient(self.pos)
            #RMS term
            self.RMSMoment = self.forgettingfactorRMS * self.RMSMoment + (1 - self.forgettingfactorRMS) * ((self.lossObj.evaluate_gradient(self.pos))**2)

            #Moment bias correction
            self.biasCorrectedMoment = self.moment / (1 - (self.decayFactor ** iteration))

            #RMS bias correction
            self.biasCorrectedRMSM = self.RMSMoment / (1 - (self.forgettingfactorRMS ** iteration))

            #Taking the step
            self.pos = self.pos - self.learningRate * (self.biasCorrectedMoment) / (np.sqrt(self.biasCorrectedRMSM) + self.epsilon)


            return self.pos
