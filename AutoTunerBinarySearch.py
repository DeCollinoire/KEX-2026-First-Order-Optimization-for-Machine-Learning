import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Project files
from QuadraticForm import QuadraticForm
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam
from utils import plotHistoryGraph, train
#from DataLoader import loadDataAsNumpyArray
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock

class autoTuneBinSearcher:
    def __init__(self, maxOptList, minOptList): # maxOpt: Optimizer, minOpt: Optimizer, maxOptList = None, minOptList = None):
        # With maximum and minimum for the search interval.
        #self.maxOpt = maxOpt
        #self.minOpt = minOpt

        self.maxOptList = maxOptList
        self.minOptList = minOptList

        self.allOptList = maxOptList + minOptList

        self.maxDict = maxOptList[0].getHyperparamDict()
        self.minDict = minOptList[0].getHyperparamDict()


#Can be improved by making new boundries truly encapsulate a better interval.
    def autoTuneBinarySearch(self, keyToAtrDict, nrEpochs=100, iterations = 10):
        #Does a binary search and returns the latest updated optimizer

        #oldBestDict = {}
        #oldBestLoss = None
        currentAnswer = [False]
        for i in range(iterations):
            #self.minOpt.reset() #***

            for minOpt in self.minOptList:
                minOpt.reset()

            #self.maxOpt.reset() #***

            for maxOpt in self.maxOptList:
                maxOpt.reset()

            self.maxDict = self.maxOptList[0].getHyperparamDict()
            self.minDict = self.minOptList[0].getHyperparamDict()

            train(self.allOptList, nrEpochs=nrEpochs)

            meanMaxOptLoss = calculateMeanFinalLoss(self.maxOptList)
            meanMinOptLoss = calculateMeanFinalLoss(self.minOptList)

            if meanMaxOptLoss >= meanMinOptLoss:

                for maxOpt in self.maxOptList:
                    setattr(maxOpt, keyToAtrDict, ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict])

                #setattr(self.maxOpt, keyToAtrDict, ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict])  # ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict]
                currentAnswer[0] = self.maxOptList[0]
            else:

                for minOpt in self.minOptList:
                    setattr(minOpt, keyToAtrDict, ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict])  # self.minDict[keyToAtrDict] = ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict]
                currentAnswer[0] = self.minOptList[0]

        return currentAnswer[0]

        # currentAnswer = [False]
        # for i in range(iterations):
        #     self.minOpt.reset()
        #     self.maxOpt.reset()
        #
        #     self.maxDict = self.maxOpt.getHyperparamDict()
        #     self.minDict = self.minOpt.getHyperparamDict()
        #
        #     train([self.maxOpt, self.minOpt], nrEpochs=nrEpochs)
        #
        #     if self.maxOpt.lossHistory[-1] >= self.minOpt.lossHistory[-1]:
        #         setattr(self.maxOpt, keyToAtrDict, ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict]) #((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict]
        #         currentAnswer[0] = self.maxOpt
        #     else:
        #         setattr(self.minOpt, keyToAtrDict, ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict])#self.minDict[keyToAtrDict] = ((self.maxDict[keyToAtrDict] - self.minDict[keyToAtrDict]) / 2) + self.minDict[keyToAtrDict]
        #         currentAnswer[0] = self.minOpt

        # return currentAnswer[0]


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

def calculateMeanFinalLoss(optList):
    sumLoss = 0
    for opt in optList:
        sumLoss = sumLoss + opt.lossHistory[-1]
    return sumLoss / len(optList)

def outerSearchLoop(keyattributeList, binSearcher: autoTuneBinSearcher, epochs=25, binarySearchIterations = 10, cycle = 1):
    resultIngOptimizerHolder = [None]
    for i in range(cycle):
        for atrStr in keyattributeList:
            resultIngOptimizerHolder[0] = binSearcher.autoTuneBinarySearch(atrStr, nrEpochs=epochs, iterations=binarySearchIterations)

    # print(resultIngOptimizerHolder[0].getHyperparamDict())
    return resultIngOptimizerHolder[0]


def optimizeHypeparamAdam(lossObj, initPosList, keyattributeList):
    # Setup loss object
    #lossObj = QuadraticForm()  # Random positive definite QDF
    #initPos = np.array([5.0, 4.0])

    maxOptList = []
    minOptList = []
    for pos in initPosList:
        maxOptList.append(adam.Adam(lossObject=lossObj, initPos=pos, learningRate=10, forgettingFactorM=0.9, forgettingFactorR=0.999))
        minOptList.append(adam.Adam(lossObject=lossObj, initPos=pos, learningRate=0.001, forgettingFactorM=0.9, forgettingFactorR=0.999))
    #optMaxAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=1, forgettingFactorM=0.9, forgettingFactorR=0.999)
    #optMinAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=0.001, forgettingFactorM=0.9, forgettingFactorR=0.999)

    binSearcher = autoTuneBinSearcher(maxOptList=maxOptList, minOptList=minOptList)

    #keyattributeList = ["lr"]


    epochs = 25
    binarySearchIterations = 100
    cycle = 1  #Cycles not working entirely as intended yet
    resultIngOptimizerHolder = [None]
    # for i in range(cycle):
    #     for atrStr in keyattributeList:
    #         resultIngOptimizerHolder[0] = binSearcher.autoTuneBinarySearch(atrStr, nrEpochs=epochs, iterations=binarySearchIterations)

    #print(resultIngOptimizerHolder[0].getHyperparamDict())


    resultIngOptimizerHolder[0] = outerSearchLoop(keyattributeList=keyattributeList, binSearcher=binSearcher, epochs=epochs, binarySearchIterations=binarySearchIterations)
    return resultIngOptimizerHolder[0]


def optimizeHypeparamMomentum(lossObj, initPosList, keyattributeList):
    # Setup loss object
    #lossObj = QuadraticForm()  # Random positive definite QDF
    #initPos = np.array([5.0, 4.0])

    maxOptList = []
    minOptList = []
    for pos in initPosList:
        maxOptList.append(momentum.Momentum(lossObject=lossObj, initPos=pos, learningRate=10, decayFactor=0.9))
        minOptList.append(momentum.Momentum(lossObject=lossObj, initPos=pos, learningRate=0.001, decayFactor=0.9))
    #optMaxAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=1, forgettingFactorM=0.9, forgettingFactorR=0.999)
    #optMinAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=0.001, forgettingFactorM=0.9, forgettingFactorR=0.999)

    binSearcher = autoTuneBinSearcher(maxOptList=maxOptList, minOptList=minOptList)

    #keyattributeList = ["lr"]


    epochs = 25
    binarySearchIterations = 100
    cycle = 1  #Cycles not working entirely as intended yet
    resultIngOptimizerHolder = [None]
    # for i in range(cycle):
    #     for atrStr in keyattributeList:
    #         resultIngOptimizerHolder[0] = binSearcher.autoTuneBinarySearch(atrStr, nrEpochs=epochs, iterations=binarySearchIterations)

    #print(resultIngOptimizerHolder[0].getHyperparamDict())


    resultIngOptimizerHolder[0] = outerSearchLoop(keyattributeList=keyattributeList, binSearcher=binSearcher, epochs=epochs, binarySearchIterations=binarySearchIterations)
    return resultIngOptimizerHolder[0]


def optimizeHypeparamNesterov(lossObj, initPosList, keyattributeList):
    # Setup loss object
    #lossObj = QuadraticForm()  # Random positive definite QDF
    #initPos = np.array([5.0, 4.0])

    maxOptList = []
    minOptList = []
    for pos in initPosList:
        maxOptList.append(nesterov.Nesterov(lossObj=lossObj, initPos=pos, lr=10, decayFactor=0.1))
        minOptList.append(nesterov.Nesterov(lossObj=lossObj, initPos=pos, lr=0.001, decayFactor=0.1))
    #optMaxAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=1, forgettingFactorM=0.9, forgettingFactorR=0.999)
    #optMinAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=0.001, forgettingFactorM=0.9, forgettingFactorR=0.999)

    binSearcher = autoTuneBinSearcher(maxOptList=maxOptList, minOptList=minOptList)

    #keyattributeList = ["lr"]


    epochs = 25
    binarySearchIterations = 100
    cycle = 1  #Cycles not working entirely as intended yet
    resultIngOptimizerHolder = [None]
    # for i in range(cycle):
    #     for atrStr in keyattributeList:
    #         resultIngOptimizerHolder[0] = binSearcher.autoTuneBinarySearch(atrStr, nrEpochs=epochs, iterations=binarySearchIterations)

    #print(resultIngOptimizerHolder[0].getHyperparamDict())


    resultIngOptimizerHolder[0] = outerSearchLoop(keyattributeList=keyattributeList, binSearcher=binSearcher, epochs=epochs, binarySearchIterations=binarySearchIterations)
    return resultIngOptimizerHolder[0]

def optimizeHypeparamSGD(lossObj, initPosList, keyattributeList):
    # Setup loss object
    #lossObj = QuadraticForm()  # Random positive definite QDF
    #initPos = np.array([5.0, 4.0])

    maxOptList = []
    minOptList = []
    for pos in initPosList:
        maxOptList.append(sgd.SGD(lossObj=lossObj, initPos=pos, lr=10))
        minOptList.append(sgd.SGD(lossObj=lossObj, initPos=pos, lr=0.001))
    #optMaxAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=1, forgettingFactorM=0.9, forgettingFactorR=0.999)
    #optMinAdam = adam.Adam(lossObject=lossObj, initPos=initPos, learningRate=0.001, forgettingFactorM=0.9, forgettingFactorR=0.999)

    binSearcher = autoTuneBinSearcher(maxOptList=maxOptList, minOptList=minOptList)

    #keyattributeList = ["lr"]


    epochs = 25
    binarySearchIterations = 100
    cycle = 1  #Cycles not working entirely as intended yet
    resultIngOptimizerHolder = [None]
    # for i in range(cycle):
    #     for atrStr in keyattributeList:
    #         resultIngOptimizerHolder[0] = binSearcher.autoTuneBinarySearch(atrStr, nrEpochs=epochs, iterations=binarySearchIterations)

    #print(resultIngOptimizerHolder[0].getHyperparamDict())


    resultIngOptimizerHolder[0] = outerSearchLoop(keyattributeList=keyattributeList, binSearcher=binSearcher, epochs=epochs, binarySearchIterations=binarySearchIterations)
    return resultIngOptimizerHolder[0]


def main():


    # Setup loss object
    lossObj = QuadraticForm()  # Random positive definite QDF
    initPosList = [np.array([5.0, 4.0])]

    # Logistic Regression
    # X, y = loadDataAsNumpyArray("datasets/australian_scale")
    # lossObj = LogisticRegression(data=[X,y])
    # initPosList = [np.random.uniform(-10, 10, lossObj.xDataLength) for _ in range(1)] # if the same position is wanted, set the seed using: np.random.seed(...)

    # Rosenbrock
    #lossObj = Rosenbrock(10)
    #initPosList = [np.array([5.0, 4.0])]

    tunedAdam = optimizeHypeparamAdam(lossObj=lossObj, initPosList=initPosList, keyattributeList=["learningRate"])
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