import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List

# Project files
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam

from QuadraticForm import QuadraticForm
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock

from utils import plotHistoryGraph, train, setupProblem
from DataLoader import loadDataAsNumpyArray

def testRobustness(listOfOptimizerListBatchVariants, batchSizeTestValues: List[int]):
    """
    To test robustness w.r.t. batch sizes, we test the (tuned) optimizers for each batch size. 
    """
    # Get the loss object
    #lossObj = optimizerList[0].lossObj

    # For every batch size test value, change the batch size and run all optimizers in parallel (using the same random batches)
    results = dict()
    for batchVariant in listOfOptimizerListBatchVariants:
        # Reset all optimizer histories,  NOTE: For some reason, NOT resetting the history makes the histories reset correctly
        # for optimizer in optimizerList:
        #     optimizer.reset()


        # Train all optimizers in parallel, using the new batch size
        #lossObj.batchSize = batchSize
        train(batchVariant)

        # Save results as a copy of each optimizer
        #optimizerListCopy = [deepcopy(optimizer) for optimizer in optimizerList]
        results[batchVariant[0].lossObj.batchSize] = batchVariant

    return results


def setupOptimizerList(lossObjList):


    optBatchSizeList = []
    optAdamList = []
    optNesterov = []
    for lossobj in lossObjList:
        privateAdam = adam.Adam(lossObject=lossobj, initPos=1, learningRate=10, forgettingFactorM=0.9, forgettingFactorR=0.999)
        optAdamList.append(privateAdam) #(adam.Adam(lossObject=lossobj, initPos=1, learningRate=10, forgettingFactorM=0.9, forgettingFactorR=0.999))

        optBatchSizeList.append([privateAdam])

    listofListsofOpt = []
    listofListsofOpt.append(optAdamList)
    listofAllOpt = []
    listofAllOpt.extend(optAdamList)
    return [listofListsofOpt, listofAllOpt]

def main():
    np.random.seed(10)
    lossObjData, initPos = setupProblem("LogReg", datasetFilepath="datasets/australian_scaled")  # australian_scaled, australian, rcv1_train.binary
    #lossObj, initPos = setupProblem("LogReg", datasetFilepath="datasets/rcv1_train.binary")

    X, y = loadDataAsNumpyArray("datasets/rcv1_train.binary")  # rcv1_train.binary or australian_scale
    # lossObj = LogisticRegression(data=[X, y])  # QuadraticForm(), Rosenbrock()


    batchSizeTestValues = [1, 32, 128, 256]
    lossObjList = []
    for batchSize in batchSizeTestValues:
        lossObjList.append(LogisticRegression(data=[X, y], batchSize=batchSize))


    # Setup base case optimizers
    organizedListVariantsOfOpt = setupOptimizerList(lossObjList=lossObjList)
    # optSGD = sgd.SGD(lossObj, initPos, lr=0.1)
    # optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    # optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
    # optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)

    # Run the test
    #batchSizeTestValues = [1, 32, 128, 256]
    #optimizerList = [optSGD, optNesterov, optMomentum, optAdam]
    results = testRobustness(organizedListVariantsOfOpt[0], batchSizeTestValues)

    # Present
    i = 0
    plt.figure()
    for batchSize, optimizerListCopies in results.items():
        # Plotting: Put all optimizers of the same batch size in the same plot 
        plt.subplot(len(batchSizeTestValues)//2+1, 2, i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.9)
        for optimizer in optimizerListCopies:
            plotHistoryGraph(optimizer.lossHistory, title = f"Loss for {optimizer.__class__.__name__}, batchSize = {batchSize}", label=f"{optimizer.__class__.__name__}, {optimizer.getHyperparamStr()}", ylabel="Loss")
        i += 1
    plt.show()

if __name__ == "__main__":
    main()