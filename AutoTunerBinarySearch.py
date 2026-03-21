import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Project files
from QuadraticForm import QuadraticForm
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam
from utils import plotHistoryGraph, train


class autoTuneBinSearcher:
    def __init__(self, maxOpt: Optimizer, minOpt: Optimizer):
        # With maximum and minimum for the search interval.
        self.maxOpt = maxOpt
        self.minOpt = minOpt

        self.maxDict = maxOpt.getHyperparamDict()
        self.minDict = minOpt.getHyperparamDict()



    def autoTuneBinarySearch(self, keyToAtrDict, nrEpochs=100, iterations = 10):
        #Does a binary search and returns the latest updated optimizer

        currentAnswer = [False]
        for i in range(iterations):
            self.minOpt.reset()
            self.maxOpt.reset()

            self.maxDict = self.maxOpt.getHyperparamDict()
            self.minDict = self.minOpt.getHyperparamDict()

            train([self.maxOpt, self.minOpt], nrEpochs=nrEpochs)

            if self.maxOpt.lossHistory[-1] >= self.minOpt.lossHistory[-1]:
                setattr(self.maxOpt, keyToAtrDict, ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict]) #((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict]
                currentAnswer[0] = self.maxOpt
            else:
                setattr(self.minOpt, keyToAtrDict, ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict])#self.minDict[keyToAtrDict] = ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict]
                currentAnswer[0] = self.minOpt

        return currentAnswer[0]


        # if lossObj is None:
        #     lossObj = optimizer.lossObj
        # # Start testing
        # for epoch in range(1, nrEpochs + 1):
        #     #Shuffle batches
        #     lossObj.fillRandomBatchList()
        #
        #
        #     lossObj.currentBatchIndex = 0
        #     for batch in range(lossObj.numberOfBatches):
        #
        #         #Store history first (to include the inital position)
        #         optimizer.posHistory.append(optimizer.pos.copy())
        #         optimizer.lossHistory.append(lossObj.evaluate_loss(optimizer.pos))
        #
        #         optimizer.step()
        #
        #         #On to next batch for calculating gradient and so on
        #         lossObj.currentBatchIndex = lossObj.currentBatchIndex + 1
        # return

def optimizeHypeparamAdam(lossObj, initPos, keyattributeList):
    # Setup loss object
    #lossObj = QuadraticForm()  # Random positive definite QDF
    #initPos = np.array([5.0, 4.0])

    optMaxAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=1, forgettingFactorM=0.9, forgettingFactorR=0.999)
    optMinAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=0.001, forgettingFactorM=0.9, forgettingFactorR=0.999)

    binSearcher = autoTuneBinSearcher(maxOpt=optMaxAdam, minOpt=optMinAdam)

    #keyattributeList = ["lr"]

    epochs = 100
    binarySearchIterations = 100
    cycle = 1
    resultIngOptimizerHolder = [None]
    for i in range(cycle):
        for atrStr in keyattributeList:
            resultIngOptimizerHolder[0] = binSearcher.autoTuneBinarySearch(atrStr, nrEpochs=epochs, iterations=binarySearchIterations)

    #print(resultIngOptimizerHolder[0].getHyperparamDict())
    return resultIngOptimizerHolder[0]


def main():


    #Setup loss object
    lossObj = QuadraticForm()  # Random positive definite QDF
    initPos = np.array([5.0, 4.0])

    tunedAdam = optimizeHypeparamAdam(lossObj=lossObj, initPos=initPos, keyattributeList=["learningRate"])
    print(tunedAdam.getHyperparamStr())
    # #Setup loss object
    # lossObj = QuadraticForm()  # Random positive definite QDF
    # initPos = np.array([5.0, 4.0])
    #
    # optMaxAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=1, forgettingFactorM=0.99999, forgettingFactorR=0.99999)
    # optMinAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=1, forgettingFactorM=0.99999, forgettingFactorR=0.99999)
    #
    # binSearcher = autoTuneBinSearcher(maxOpt=optMaxAdam, minOpt=optMinAdam)
    #
    # keyattributeList = ["lr"]
    #
    # cycle = 1
    #
    #
    # resultIngOptimizerHolder = [None]
    # for i in range(cycle):
    #     for atrStr in keyattributeList:
    #         resultIngOptimizerHolder[0] = binSearcher.autoTuneBinarySearch(atrStr, 100, 10)
    #
    # print(resultIngOptimizerHolder[0].getHyperparamDict())

if __name__ == "__main__":
    main()