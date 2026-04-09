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

def testRobustness(optimizerList: List[Optimizer], batchSizeTestValues: List[int], nrEpochs=50):
    """
    To test robustness w.r.t. batch sizes, we test the (tuned) optimizers for each batch size. 
    """
    # Get the loss object
    lossObj = optimizerList[0].lossObj

    # For every batch size test value, change the batch size and run all optimizers in parallel (using the same random batches)
    results = dict()
    for batchSize in batchSizeTestValues:
        print(f"Testing batchSize = {batchSize}")
        # Reset all optimizer histories,  NOTE: For some reason, NOT resetting the history makes the histories reset correctly
        for optimizer in optimizerList:
            optimizer.reset()

        # Train all optimizers in parallel, using the new batch size
        lossObj.batchSize = batchSize
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
    #np.random.seed(10)
    lossObj, initPos = setupProblem("LogReg", datasetFilepath="datasets/australian_scaled")  # australian_scaled, australian, rcv1_train.binary

    # Setup base case optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr=0.1)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)

    # Run the test
    batchSizeTestValues = [1, 16, 32, 64, 256, lossObj.xDataListLength] # lossObj.xDataListLength for full batch
    optimizerList = [optSGD, optNesterov, optMomentum, optAdam]
    results = testRobustness(optimizerList, batchSizeTestValues, nrEpochs=10)

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
    plt.savefig("images/all_optimizers_robustness_test.png", dpi=300)   # TODO: Fix legends
    plt.show()

if __name__ == "__main__":
    main()