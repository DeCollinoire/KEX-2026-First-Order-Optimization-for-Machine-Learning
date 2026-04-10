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
    results = dict()
    for batchVariant in listOfOptimizerListBatchVariants:
        train(batchVariant)
        results[batchVariant[0].lossObj.batchSize] = batchVariant
    return results


def setupOptimizerList(lossObjList, initPos):
    optAdamList = []
    optSGDList = []
    listOfAllOpt = []

    groupedByOptimizer = [] # for presenting
    groupedByBatches = [] # for training
    for lossobj in lossObjList:
        privateSGD = sgd.SGD(lossobj, initPos, lr=0.1)
        privateAdam = adam.Adam(lossObject=lossobj, initPos=initPos, learningRate=0.5, forgettingFactorM=0.9, forgettingFactorR=0.999)
        
        optSGDList.append(privateSGD)
        optAdamList.append(privateAdam)
        
        # All optimizers of same batch size
        groupedByBatches.append([privateSGD, privateAdam])

    # All optimizers of same size
    groupedByOptimizer.append([optSGDList, optAdamList])
    listOfAllOpt.extend(optSGDList)
    listOfAllOpt.extend(optAdamList)
    return groupedByOptimizer, groupedByBatches, listOfAllOpt

def main():
    # Setup lossObj
    X, y = loadDataAsNumpyArray("datasets/rcv1_train.binary")  # rcv1_train.binary or australian_scale
    batchSizeTestValues = [1, 32, 128, 256]
    lossObjList = []
    for batchSize in batchSizeTestValues:
        lossObjList.append(LogisticRegression(data=[X, y], batchSize=batchSize))

    # Setup base case optimizers
    np.random.seed(10)
    initPos = np.random.uniform(-10, 10, lossObjList[0].xDataLength)
    groupedByOptimizer, groupedByBatches, _ = setupOptimizerList(lossObjList=lossObjList, initPos=initPos)

    # Run the test
    results = testRobustness(groupedByBatches, batchSizeTestValues)

    # Present
    # TODO: Change to plot by optimizer variant
    i = 0
    plt.figure()
    for batchSize, optimizerListCopies in results.items():
        # Plotting: Put all optimizers of the same batch size in the same plot 
        plt.subplot(len(batchSizeTestValues)//2+1, 2, i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.9)
        for optimizer in optimizerListCopies:
            plotHistoryGraph(optimizer.lossHistory, title = f"Loss history, batchSize = {batchSize}", label=f"{optimizer.__class__.__name__}, {optimizer.getHyperparamStr()}", ylabel="Loss")
        i += 1
    plt.show()

if __name__ == "__main__":
    main()