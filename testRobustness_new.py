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

def setupOptimizerList(lossObjList, initPos, problemName="rcv1"):
    # Optimizer config contains the tuned optimizers
    optimizerConfig = {
        "Rosenbrock": {
            "SGD": {"lr": 0.0008},
            "Momentum": {"learningRate": 0.002, "decayFactor": 0.9},
            "Nesterov": {"lr": 0.005, "decayFactor": 0.92},
            "Adam": {"learningRate": 0.0168, "forgettingFactorM": 0.95, "forgettingFactorR": 0.999}
        },
        "australian": { # these are assumed to be the same as for australian_scale
            "SGD": {"lr": 0.04},
            "Momentum": {"learningRate": 0.03, "decayFactor": 0.5},
            "Nesterov": {"lr": 0.035, "decayFactor": 0.5},
            "Adam": {"learningRate": 0.25, "forgettingFactorM": 0.93, "forgettingFactorR": 0.999}
        },
        "australian_scale": {
            "SGD": {"lr": 0.04},
            "Momentum": {"learningRate": 0.03, "decayFactor": 0.5},
            "Nesterov": {"lr": 0.035, "decayFactor": 0.5},
            "Adam": {"learningRate": 0.25, "forgettingFactorM": 0.93, "forgettingFactorR": 0.999}
        },
        "rcv1": {
            "SGD": {"lr": 0.07},
            "Momentum": {"learningRate": 0.07, "decayFactor": 0.8},
            "Nesterov": {"lr": 0.07, "decayFactor": 0.8},
            "Adam": {"learningRate": 0.3, "forgettingFactorM": 0.92, "forgettingFactorR": 0.99}
        }
    }
    print(f"Setting up optimizers for {problemName}, initPos: {initPos}")
    optSGDList = []
    optMomentumList = []
    optNesterovList = []
    optAdamList = []
    groupedByOptimizer = [] # for presenting grouped by optimizer
    groupedByBatches = [] # for training grouped by batch size

    # Create lossObjects
    for lossObj in lossObjList:
        privateSGD = sgd.SGD(lossObj, initPos, **optimizerConfig[problemName]["SGD"])
        privateMomentum = momentum.Momentum(lossObj, initPos, **optimizerConfig[problemName]["Momentum"])
        privateNesterov = nesterov.Nesterov(lossObj, initPos, **optimizerConfig[problemName]["Nesterov"])
        privateAdam = adam.Adam(lossObj, initPos, **optimizerConfig[problemName]["Adam"])

        optSGDList.append(privateSGD)
        optMomentumList.append(privateMomentum)
        optNesterovList.append(privateNesterov)
        optAdamList.append(privateAdam)
        
        # All optimizers of same batch size
        groupedByBatches.append([privateSGD, privateMomentum, privateNesterov, privateAdam])

    # All optimizers of same size
    groupedByOptimizer.extend([optSGDList, optMomentumList, optNesterovList, optAdamList])

    return groupedByOptimizer, groupedByBatches

datasetMap = {
    "australian": "datasets/australian",
    "australian_scale": "datasets/australian_scale",
    "rcv1": "datasets/rcv1_train.binary"
}

batchSizeConfig = {
    "australian": [1, 32, 128, 512, 690],
    "australian_scale": [1, 16, 64, 128, 256, 512, 690],
    "rcv1": [32, 128, 512, 1024, 4096, 20242] # 20242 is the total number of samples, so this is the full batch case
}

def main():
    # Config
    randomSeed = 25
    problemName = "australian_scale"
    datasetFilepath = datasetMap.get(problemName, "N/A")
    l2NormalizationOn = (problemName in ["australian", "australian_scale"])
    initialPosInterval = 0
    nrEpochs = 10
    batchSizeTestValues = batchSizeConfig.get(problemName, [32, 128, 512, 1024])

    # Setup lossObj
    np.random.seed(randomSeed)
    X, y = loadDataAsNumpyArray(datasetFilepath, toDense=False, l2NormalizationOn=l2NormalizationOn)  # rcv1_train.binary or australian_scale. X and y are sparse matrices, but will be converted to dense in the setupProblem function if 'toDense = True' is set.
    nrSamples, nrFeatures = X.shape # type: ignore - used to set relative batch sizes
    batchSizeTestValues = [np.minimum(batchSize, nrSamples) for batchSize in batchSizeTestValues]   # Set batch size to <= nrSamples
    lossObjList = []
    for batchSize in batchSizeTestValues:
        lossObjList.append(LogisticRegression(data=[X, y], batchSize=batchSize))
    print("Loss objects are set up for batch sizes: ", batchSizeTestValues)

    # Setup base case optimizers
    initPos = np.random.uniform(-initialPosInterval, initialPosInterval, size=nrFeatures)
    groupedByOptimizer, groupedByBatches = setupOptimizerList(lossObjList=lossObjList, initPos=initPos, problemName=problemName)
    print("Optimizers set up finished.")

    # Run the test
    print("Starting robustness test..."); startTime = default_timer()
    testRobustness(groupedByBatches, nrOfEpochs=nrEpochs)
    print(f"Robustness test completed in {default_timer() - startTime:.2f} seconds.")

    # Present
    i = 0
    for optimizerList in groupedByOptimizer:
        # Plotting: Put all optimizers of the same batch size in the same plot 
        plt.figure(figsize=(17, 10))
        # plt.subplot(2, 2, i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.9)
        for optimizer in optimizerList:
            plotHistoryGraph(optimizer.lossHistory, 
                             title = f"Robustness test, loss history for {optimizer.__class__.__name__}, lossObj = {optimizer.lossObj.__class__.__name__}, dataset = {datasetFilepath}, randomSeed = {randomSeed}", 
                             label = f"{optimizer.__class__.__name__}, {optimizer.getHyperparamStr()}, batchSize = {optimizer.lossObj.batchSize}", 
                             ylabel = "Loss",
                             marker=""
                             )
            plt.minorticks_on()
            plt.grid(True, which="minor", linestyle=":", linewidth=1)
        i += 1
        plt.savefig(f"images/robustness_test_results_{problemName}_{optimizerList[0].__class__.__name__}.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()