from timeit import default_timer

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

def testRobustness(groupedByBatches, nrOfEpochs=20):
    """
    To test robustness w.r.t. batch sizes, we test the (tuned) optimizers for each batch size. 
    """
    for batchVariant in groupedByBatches:
        print("Testing for batch size: ", batchVariant[0].lossObj.batchSize)
        train(batchVariant, nrEpochs=nrOfEpochs)

def setupOptimizerList(lossObjList, initPos):
    optSGDList = []
    optMomentumList = []
    optNesterovList = []
    optAdamList = []
    groupedByOptimizer = [] # for presenting grouped by optimizer
    groupedByBatches = [] # for training grouped by batch size

    # Create lossObjects
    for lossObj in lossObjList:
        privateSGD = sgd.SGD(lossObj, initPos, lr=0.1)
        privateMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
        privateNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
        privateAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)
        
        optSGDList.append(privateSGD)
        optMomentumList.append(privateMomentum)
        optNesterovList.append(privateNesterov)
        optAdamList.append(privateAdam)
        
        # All optimizers of same batch size
        groupedByBatches.append([privateSGD, privateMomentum, privateNesterov, privateAdam])

    # All optimizers of same size
    groupedByOptimizer.extend([optSGDList, optMomentumList, optNesterovList, optAdamList])

    return groupedByOptimizer, groupedByBatches

def main():
    # Setup lossObj
    # Load using loadDataAsNumpyArray, because the setupProblem returns lossObj
    datasetFilepath = "datasets/rcv1_train.binary" # rcv1_train.binary or australian_scale
    X, y = loadDataAsNumpyArray(datasetFilepath, toDense=False)  # rcv1_train.binary or australian_scale. X and y are sparse matrices, but will be converted to dense in the setupProblem function if 'toDense = True' is set.

    # Batch size values to test, relative to number of samples
    nrOfSamples = X.shape[0]
    batchSizeTestValues = [1, 128, 512, 1024] # [round(factor*nrOfSamples) for factor in [0.1, 0.25, 0.5, 1.0]]
    
    # Create lossObj
    lossObjList = []
    for batchSize in batchSizeTestValues:
        lossObjList.append(LogisticRegression(data=[X, y], batchSize=batchSize))
    print("Loss objects set up for batch sizes: ", batchSizeTestValues)

    # Setup base case optimizers
    np.random.seed(10)
    initPos = np.random.uniform(-0.1, 0.1, lossObjList[0].xDataLength)
    groupedByOptimizer, groupedByBatches = setupOptimizerList(lossObjList=lossObjList, initPos=initPos)
    print("Optimizers set up.")

    # Run the test
    print("Starting robustness test...")
    startTime = default_timer()
    testRobustness(groupedByBatches, nrOfEpochs=1)
    print(f"Robustness test completed in {default_timer() - startTime:.2f} seconds.")

    # Present
    i = 0
    plt.figure()
    for optimizerList in groupedByOptimizer:
        # Plotting: Put all optimizers of the same batch size in the same plot 
        plt.subplot(len(batchSizeTestValues)//2+1, 2, i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.9)
        for optimizer in optimizerList:
            plotHistoryGraph(optimizer.lossHistory, title = f"Loss history, lossObj = {optimizer.lossObj.__class__.__name__}, dataset = {datasetFilepath}", label=f"{optimizer.__class__.__name__}, {optimizer.getHyperparamStr()}, batchSize = {optimizer.lossObj.batchSize}", ylabel="Loss")
        i += 1
    plt.show()

if __name__ == "__main__":
    main()