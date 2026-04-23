import numpy as np
from copy import deepcopy

class Optimizer:
    def __init__(self, initPos = np.array([])):
        # Store parameters/position and the lossObj
        self.initPos = np.array(deepcopy(initPos), dtype=float)
        self.pos = np.array(deepcopy(initPos), dtype=float)
        self.lossHistory = []
        self.posHistory = []

    def savePosition(self, lossObj):
        currentPos = self.pos.copy()
        self.posHistory.append(currentPos) # type: ignore
        self.lossHistory.append(lossObj.evaluate_loss(currentPos))

    def getHyperparamStr(self):
        return str()
    
    def getHyperparamDict(self):
        return {}

    def step(self, lossObj):
        return np.array([])

    def resetHistory(self):
        self.posHistory = []
        self.lossHistory = []

    def resetPosition(self):
        self.pos = deepcopy(self.initPos)

    def reset(self):
        self.resetHistory()
        self.resetPosition()

    def preAllocateHistory(self, amountOfSteps):
        self.posHistory = [0 for _ in range(amountOfSteps)]
        self.lossHistory = [0 for _ in range(amountOfSteps)]

class SGD(Optimizer):
    def __init__(self, initPos, lr):
        super().__init__(initPos)
        self.lr = lr

    def getHyperparamStr(self):
        return f"lr: {self.lr}"

    def getHyperparamDict(self):
        return {"lr": self.lr}

    def step(self, lossObj):
        grad = lossObj.evaluate_gradient(self.pos)
        self.pos -= self.lr * grad
        return self.pos

class Nesterov(Optimizer):
    def __init__(self, initPos, lr = 0.1, decayFactor = 0.1):
        super().__init__(initPos)
        # Hyperparameters
        self.lr = lr
        self.decayFactor = decayFactor
        
        # Velocity vector
        self.velocity = np.zeros_like(initPos)

    def getHyperparamStr(self):
        return f"lr: {self.lr}, decayFactor: {self.decayFactor}"

    def getHyperparamDict(self):
        return {"lr": self.lr, "decayFactor": self.decayFactor}

    def step(self, lossObj):
        look_ahead_pos = self.pos - self.decayFactor * self.velocity
        self.velocity = self.decayFactor * self.velocity + self.lr * lossObj.evaluate_gradient(look_ahead_pos)    # estimate the gradient at the next position
        self.pos -= self.velocity
        return self.pos

    def reset(self):
        super().reset()

        # Velocity vector
        self.velocity = np.zeros_like(self.initPos)

class Momentum(Optimizer):
    def __init__(self, initPos, lr, decayFactor):
        super().__init__(initPos)
        self.lr = lr
        self.decayFactor = decayFactor

        #Moment/velocity term
        self.moment = np.zeros_like(initPos)

    def step(self, lossObj):
        self.moment = self.decayFactor * self.moment + self.lr * lossObj.evaluate_gradient(self.pos)
        self.pos = self.pos - self.moment
        return self.pos

    def getHyperparamStr(self):
        return f"lr: {self.lr}, decayfactor: {self.decayFactor}"

    def getHyperparamDict(self):
        return {"lr": self.lr, "decayFactor": self.decayFactor}

    def reset(self):
        super().reset()
        
        # Moment/velocity term
        self.moment = np.zeros_like(self.initPos)

class Adam(Optimizer):
    def __init__(self, initPos, lr, beta1 = 0.9, beta2 = 0.999):
        """
        The beta1 is for the momentum part in adam while the other forgetting factor is for the RMSprop part
        :param lossObject:
        :param beta1:
        :param beta2:
        :param lr:
        """
        super().__init__(initPos)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # Moment/velocity term
        self.moment = np.zeros_like(initPos)

        # For the second moment in adam, for the rms part
        self.RMSMoment = np.zeros_like(initPos)

        # For avoiding zero division in adam
        self.epsilon = 1e-8

        # Number of iterations/steps taken
        self.iteration = 1

    def step(self, lossObj):
        grad = lossObj.evaluate_gradient(self.pos)

        # Moment term
        self.moment = self.beta1 * self.moment + (1 - self.beta1) * grad

        # RMS term
        self.RMSMoment = self.beta2 * self.RMSMoment + (1 - self.beta2) * (grad**2)

        # Moment bias correction
        biasCorrectedMoment = self.moment / (1 - (self.beta1 ** self.iteration))

        # RMS bias correction
        biasCorrectedRMSM = self.RMSMoment / (1 - (self.beta2 ** self.iteration))

        # Taking the step
        self.pos = self.pos - self.lr * (biasCorrectedMoment) / (np.sqrt(biasCorrectedRMSM) + self.epsilon)

        self.iteration = self.iteration + 1

        return self.pos

    def getHyperparamStr(self):
        return f"lr: {self.lr}, beta1: {self.beta1}, beta2: {self.beta2}"
    
    def getHyperparamDict(self):
        return {"lr": self.lr, "beta1": self.beta1, "beta2": self.beta2}

    def reset(self):
        super().reset()

        # Reset moments
        # Moment/velocity term
        self.moment = np.zeros_like(self.initPos)

        # For the second moment in adam, for the rms part
        self.RMSMoment = np.zeros_like(self.initPos)

        self.iteration = 1