import optimizers.optimizer as opt
import optimizers.loss.loss as lo
import numpy as np

class Adam(opt.Optimizer):
    def __init__(self, lossObject: lo.LossObj, initPos, learningRate, forgettingFactorM = 0.9, forgettingFactorR = 0.999):
        """
        The forgettingFactorM is for the momentum part in adam while the other forgetting factor is for the RMSprop part
        :param lossObject:
        :param forgettingFactorM:
        :param forgettingFactorR:
        :param learningRate:
        """
        super().__init__(lossObject, initPos)

        self.learningRate = learningRate
        self.forgettingFactorM = forgettingFactorM
        self.forgettingFactorR = forgettingFactorR

        # Moment/velocity term
        self.moment = np.zeros_like(initPos)
        # For the bias corrected term
        self.biasCorrectedMoment = np.zeros_like(initPos)

        # For the second moment in adam, for the rms part
        self.RMSMoment = np.zeros_like(initPos)
        # For the bias corrected term
        self.biasCorrectedRMSM = np.zeros_like(initPos)

        # For avoiding zero division in adam
        self.epsilon = 1e-8

        # Number of iterations/steps taken
        self.iteration = 1

    def step(self):
        # Moment term
        self.moment = self.forgettingFactorM * self.moment + (1 - self.forgettingFactorM) * self.lossObj.evaluate_gradient(self.pos)

        # RMS term
        self.RMSMoment = self.forgettingFactorR * self.RMSMoment + (1 - self.forgettingFactorR) * ((self.lossObj.evaluate_gradient(self.pos))**2)

        # Moment bias correction
        self.biasCorrectedMoment = self.moment / (1 - (self.forgettingFactorM ** self.iteration))

        # RMS bias correction
        self.biasCorrectedRMSM = self.RMSMoment / (1 - (self.forgettingFactorR ** self.iteration))

        # Taking the step
        self.pos = self.pos - self.learningRate * (self.biasCorrectedMoment) / (np.sqrt(self.biasCorrectedRMSM) + self.epsilon)

        self.iteration = self.iteration + 1

        return self.pos

    def getHyperparamStr(self):
        return f"lr: {self.learningRate}, beta1: {self.forgettingFactorM}, beta2: {self.forgettingFactorR}"
    
    def getHyperparamDict(self):
        return {"learningRate": self.learningRate, "forgettingFactorM": self.forgettingFactorM, "forgettingFactorR": self.forgettingFactorR}

    def reset(self):
        # Reset history & position
        self.resetHistory()
        self.resetPosition()

        # Reset moments
        # Moment/velocity term
        self.moment = np.zeros_like(self.initPos)
        # For the bias corrected term
        self.biasCorrectedMoment = np.zeros_like(self.initPos)

        # For the second moment in adam, for the rms part
        self.RMSMoment = np.zeros_like(self.initPos)
        # For the bias corrected term
        self.biasCorrectedRMSM = np.zeros_like(self.initPos)


        self.iteration = 1

        return
