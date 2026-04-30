import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Project files
from QuadraticForm import QuadraticForm
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam
from utils import plotHistoryGraph, train
from DataLoader import loadDataAsNumpyArray
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock
from utils import setupProblem


def testSensitivity():
    pass


def setupOptimizerList(lossObj, initPos):
    """
    Returns optimizers in the following format
    [
        # Optimizer by group, for example SGD first
        [
            # Variation in hyperparameter
            [

            ]
            [

            ]
        ]
        ...

    ]
    :param lossObj:
    :param initPos:
    :return:
    """
    optSGDList = []
    optMomentumList = []
    optNesterovList = []
    optAdamList = []
    groupedByOptimizer = []  # for presenting grouped by optimizer
    allOpts = []


    # The following lists will contain their respective hyperparameter variation optimizers
    optSGDLrList = []

    optMomentumLrList = []
    optMomentumDfList = []

    optNesterovLrList = []
    optNesterovDfList = []

    optAdamLrList = []
    optAdamFfMList = []
    optAdamFfRList = []

    # SGD setup
    SGDLrList = [0.001, 0.01, 0.05, 0.1]
    for lr in SGDLrList:
        optSGDLrList.append(sgd.SGD(lossObj, initPos, lr=lr))
    optSGDList.append(optSGDLrList)

    # Momentum setup
    momentumLrList = [0.06, 0.07, 0.075, 0.08, 0.09, 0.1]
    momentumDfList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    for lr in momentumLrList:
        optMomentumLrList.append(momentum.Momentum(lossObj, initPos, learningRate=lr, decayFactor=0.9))
    for df in momentumDfList:
        optMomentumDfList.append(momentum.Momentum(lossObj, initPos, learningRate=0.035, decayFactor=df))
    optMomentumList.append(optMomentumLrList)
    optMomentumList.append(optMomentumDfList)

    # Nesterov setup
    nesterovLrList = [0.01, 0.02, 0.03, 0.1]
    nesterovDfList = [0.1, 0.2, 0.5, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    for lr in nesterovLrList:
        optNesterovLrList.append(nesterov.Nesterov(lossObj, initPos, lr=lr, decayFactor=0.9))
    for df in nesterovDfList:
        optNesterovDfList.append(nesterov.Nesterov(lossObj, initPos, lr=0.035, decayFactor=df))
    optNesterovList.append(optNesterovLrList)
    optNesterovList.append(optNesterovDfList)

    # Adam setup
    adamBaseCaselr = 0.2
    adamBaseCaseForgettingfactorM = 0.91
    adamBaseCaseForgettingfactorR = 0.999
    adamLrList = [0.1, 0.15, 0.2, 0.3, 0.5, 1]
    adamFfMList = [0.2, 0.8, 0.9, 0.91, 0.95]
    adamFfRList = [0.7, 0.8, 0.99, 0.999]
    for lr in adamLrList:
        optAdamLrList.append(adam.Adam(lossObj, initPos, learningRate=lr, forgettingFactorM=adamBaseCaseForgettingfactorM, forgettingFactorR=adamBaseCaseForgettingfactorR))
    for ffM in adamFfMList:
        optAdamFfMList.append(adam.Adam(lossObj, initPos, learningRate=adamBaseCaselr, forgettingFactorM=ffM, forgettingFactorR=adamBaseCaseForgettingfactorR))
    for ffR in adamFfRList:
        optAdamFfRList.append(adam.Adam(lossObj, initPos, learningRate=adamBaseCaselr, forgettingFactorM=adamBaseCaseForgettingfactorM,
                                        forgettingFactorR=ffR))
    for i in [optAdamLrList, optAdamFfMList, optAdamFfRList]:
        optAdamList.append(i)


    groupedByOptimizer.extend([optSGDList, optMomentumList, optNesterovList, optAdamList])

    for optType in groupedByOptimizer:
        for optHype in optType:
            allOpts.extend(optHype)

    return groupedByOptimizer, allOpts


def main():
    # Setup problem
    datasetFilepath = "datasets/australian_scale"  # rcv1_train.binary or australian_scale
    print("Set up problem: Begun")
    problemName = "LogReg"
    lossObj, initPos = setupProblem(problemName=problemName, dim=10, datasetFilepath=datasetFilepath,
                                    initialPosInterval=0, randomSeed=10,
                                    batchSize=1000000, toDense=False, l2NormalizationOn=True)  # QDF, Rosenbrock; datasetFilepath is only needed for LogReg
    print("Set up problem: Finished")
    print("initpos" + str(initPos))

    # Rosenbrock
    # datasetFilepath = "Rosenbrock" # This is also used to show the name in the plot title
    # lossObj, initPos = setupProblem("Rosenbrock", dim=10, randomSeed = randomSeed)

    # lossObj, initPos = setupProblem("QDF", randomSeed = randomSeed)


    # Get the optimizers by type of optimizer in a list
    groupedByOptimizer, allOpts = setupOptimizerList(lossObj=lossObj, initPos=initPos)

    train(allOpts, lossObj=lossObj,nrEpochs=100)


    # Plot
    for optimizerTypeList in groupedByOptimizer:

        # Create a new window/figure
        plt.figure(figsize=(10, 8))
        num_subplots = len(optimizerTypeList)

        # Iterate through variations to create subplots
        for i, optimizerList in enumerate(optimizerTypeList):
            # Create a grid: 'num_subplots' rows, 1 column, current index (starts at 1)
            plt.subplot(num_subplots, 1, i + 1)

            # Plot all optimizers for this specific subplot
            for optimizer in optimizerList:
                plotHistoryGraph(
                    optimizer.lossHistory,
                    title=f"Loss: {optimizer.lossObj.__class__.__name__}, Dataset: {datasetFilepath}",
                    label=f"{optimizer.__class__.__name__}, {optimizer.getHyperparamStr()}",
                    ylabel="Loss"
                )

        # Clean up layout for the whole figure
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.9)

    plt.show()


if __name__ == "__main__":
    main()