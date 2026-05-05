import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List
from math import ceil

# Project files
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam

from QuadraticForm import QuadraticForm
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock

from utils import plotHistoryGraph, train, setupProblem
from DataLoader import loadDataAsNumpyArray

"""
DEPRECATED VERSION
NOTE: This version uses setBatchSize to internally change the batch size, and then assumes
"""

def testRobustness(optimizerList: List[Optimizer], batchSizeTestValues: List[int], nrEpochs=50):
    """
    To test robustness w.r.t. batch sizes, we test the (tuned) optimizers for each batch size.
    We assume all optimizers in the optimizerList have the same lossObj.
    """
    # Get the loss object
    lossObj = optimizerList[0].lossObj

    # For every batch size test value, change the batch size and run all optimizers in parallel (using the same random batches)
    results = dict()
    for batchSize in batchSizeTestValues:
        print(f"Testing batchSize = {batchSize}")
        # Reset all optimizer histories
        for optimizer in optimizerList:
            optimizer.reset()

        # Train all optimizers in parallel, using the new batch size
        lossObj.setBatchSize(batchSize)
        train(optimizerList, nrEpochs=nrEpochs)

        # Save results as a copy of each optimizer
        for optimizer in optimizerList:
            optName = optimizer.__class__.__name__
            if results.get(optName, None) is None:
                results[optName] = [deepcopy(optimizer)] # initialize result list, e.g. [SGD_1, SGD_2, ...], where each have different batch sizes 
            else:
                results[optName].append(deepcopy(optimizer))         
    return results



def main():
    lossObj, initPos = setupProblem("LogReg", 
                                    datasetFilepath="datasets/rcv1_train.binary", 
                                    randomSeed=10)  # australian_scale, australian, rcv1_train.binary
    print("Initial position: ",initPos)

    # Setup base case optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr=0.1)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.5, forgettingFactorM=0.9, forgettingFactorR=0.999)

    nrOfSamples = lossObj.xDataListLength
    batchSizeTestValues = [round(factor*nrOfSamples) for factor in [0.1, 0.25, 0.5, 1.0]]
    optimizerList = [optSGD, optNesterov, optMomentum, optAdam]

    # Run the test
    results = testRobustness(optimizerList, batchSizeTestValues, nrEpochs=25)

    # Present
    plotIdx = 1
    plt.figure(figsize=(12, 8))
    for optName, optimizerListCopies in results.items():
        # Plotting: Put all optimizers of the same batch size in the same plot
        plt.subplot(2,2, plotIdx)
        for i, optimizer in enumerate(optimizerListCopies):
            batchSize = batchSizeTestValues[i] # Want to show batch sizes as well
            plotHistoryGraph(optimizer.lossHistory, title = f"Loss for {optimizer.__class__.__name__}", label=f"{optimizer.__class__.__name__}, batchSize = {batchSize}, {optimizer.getHyperparamStr()}", ylabel="Loss")
        plotIdx += 1
    plt.savefig("images/all_optimizers_robustness_test.png", dpi=300)   # TODO: Fix legends that covers the plot
    plt.show()

if __name__ == "__main__":
    main()