import numpy as np
from optimizers.loss.loss import LossObj
from copy import deepcopy

class Optimizer:
    def __init__(self, lossObj: LossObj, initPos = np.array([])):
        # Store parameters/position and the lossObj
        self.lossObj = lossObj

        self.initPos = np.array(deepcopy(initPos), dtype=float)
        self.pos = np.array(deepcopy(initPos), dtype=float)

        self.lossHistory = []
        self.posHistory = []

    def __call__(self, nr_epochs = 100):
        return self.optimize(nr_epochs)

    def optimize(self, nr_epochs = 100):
        """ Full optimization. Loops through the entire dataset 'nr_epochs' times in batches."""
        # Reset history
        self.posHistory = [self.pos.copy()]
        self.lossHistory = [self.lossObj.evaluate_loss(self.pos)]
        
        # Step through the entire dataset nr_epochs times
        for _ in range(nr_epochs):
            # Step once for every batch
            self.lossObj.fillRandomBatchList() 
            for batchIndex in range(self.lossObj.numberOfBatches):
                self.step()
                
                # Track history
                self.posHistory.append(self.pos.copy())
                self.lossHistory.append(self.lossObj.evaluate_loss(self.pos))

                self.lossObj.currentBatchIndex += 1

        return np.array(self.posHistory), np.array(self.lossHistory)

    def savePosition(self, historyIndex=None):

        currentPos = self.pos.copy()
        if historyIndex is None:
            #currentPos = self.pos.copy()
            self.posHistory.append(currentPos)
            self.lossHistory.append(self.lossObj.evaluate_loss(currentPos))
        else:
            self.posHistory[historyIndex] = currentPos
            self.lossHistory[historyIndex] = self.lossObj.evaluate_loss(currentPos)

    def getHyperparamStr(self):
        return ""
    
    def getHyperparamDict(self):
        return {}

    def step(self):
        return np.array([])

    def resetHistory(self):
        self.posHistory = []
        self.lossHistory = []
        return

    def resetPosition(self):
        self.pos = deepcopy(self.initPos)

    def reset(self):
        pass

    def preAllocateHistory(self, amountOfSteps):
        self.posHistory = [0 for _ in range(amountOfSteps)]
        self.lossHistory = [0 for _ in range(amountOfSteps)]
