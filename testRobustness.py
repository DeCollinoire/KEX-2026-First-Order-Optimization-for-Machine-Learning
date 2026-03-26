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

def testRobustness(optimizerList: List[Optimizer], batchSizeTestValues):
    """
    To test robustness w.r.t. batch sizes, we test the (tuned) optimizers for each batch size. 
    """
    # Get the loss object
    lossObj = optimizerList[0].lossObj

    # For every batch size test value, change the batch size and run all optimizers in parallel (using the same random batches)
    results = dict()
    for batchSize in batchSizeTestValues:
        # Reset all optimizer histories,  NOTE: For some reason, NOT resetting the history makes the histories reset correctly
        #for optimizer in optimizerList:
        #    optimizer.reset()

        # Train all optimizers in parallel, using the new batch size
        lossObj.batchSize = batchSize
        train(optimizerList)

        # Save results as a copy of each optimizer
        optimizerListCopy = [deepcopy(optimizer) for optimizer in optimizerList]
        results[batchSize] = optimizerListCopy
    return results

def main():
    np.random.seed(10)
    lossObj, initPos = setupProblem("LogReg", datasetFilepath="datasets/australian_scaled")  # australian_scaled, australian, rcv1_train.binary
    # lossObj, initPos = setupProblem("LogReg", datasetFilepath="datasets/rcv1_train.binary")

    # Setup base case optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr=0.1)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)

    # Run the test
    batchSizeTestValues = [1, 4, 8, 16, 32, 64, 128, 256]
    optimizerList = [optSGD, optNesterov, optMomentum, optAdam]
    results = testRobustness(optimizerList, batchSizeTestValues)

    # Present
    i = 0
    plt.figure()
    for batchSize, optimizerListCopies in results.items():
        # Plotting: Put all optimizers of the same batch size in the same plot 
        plt.subplot(len(batchSizeTestValues)//2, 2, i+1)
        for optimizer in optimizerListCopies:
            plotHistoryGraph(optimizer.lossHistory, title = f"Loss for {optimizer.__class__.__name__}, batchSize = {batchSize}", label=f"{optimizer.__class__.__name__}, {optimizer.getHyperparamStr()}", ylabel="Loss")
        i += 1
    plt.show()

if __name__ == "__main__":    
    main()