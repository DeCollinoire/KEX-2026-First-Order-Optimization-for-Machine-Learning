import numpy as np
import matplotlib.pyplot as plt
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam
from QuadraticForm import QuadraticForm
from utils import plotHistoryGraph, train
from copy import deepcopy

"""
DEPRECATED VERSION
This version needs to add presentation, and potentially to improve the 'optimizerGroup' class.
"""

class optimizerGroup:
    def __init__(self, optimizerBaseCase: Optimizer):
        self.optimizerBaseCase = optimizerBaseCase
        self.optVariants = optimizerBaseCase.getHyperparamDict()

    def setVariantsManual(self, hyperparamName, hyperparamValues):
        """
        Creates a dictionary of variants in one hyperparamter according to this mapping:
            hyperparamName: [opt_h1, opt_h2, opt_h3]
        where opt_h1 is the basecase optimizer with the chosen hyperparameter set to hyperparamValues[1] etc.
        """
        # Setup all
        optimizerList = []
        for value in hyperparamValues:
            opt = deepcopy(self.optimizerBaseCase)
            setattr(opt, hyperparamName, value)
            optimizerList.append(opt)
        self.optVariants[hyperparamName] = optimizerList

    def trainAllVariants(self):
        # For each hyperparameter, train each adjusted optimizer
        for hyperparamName, optimizerList in self.optVariants.items():
            train(optimizerList, nrEpochs=100)

def calculateMeanDifference(base, lossHistories):
    """
    Gets the "final" loss history value for each loss history and calculates the mean between them and the base value.
    """
    pass

def testHyperparameterSensitivity(optimizerBaseCase: Optimizer, nrEpochs=100):
    """"
    Results will be a dictionary on the following format:
    optVariants = {
        "h1": [opt11, opt12] # hyperparam1
        "h1": [opt21, opt22] # hyperparam2
    }
    where opt11 is the basecase with hyperparameter "h1" adjusted
    """
    # Get all hyperparameters to be varied.
    hyperparamsDict = optimizerBaseCase.getHyperparamDict()
    optVariants = hyperparamsDict.copy() # Values will be overwritten with optimizer objects in the loop below

    # Test each hyperparameter
    adjustmentFactor = [0.001, 0.01, 0.1, 1]
    for hyperparamName, baseval in hyperparamsDict.items():
        # Vary the value of the hyperparameter
        optVariants[hyperparamName] = []
        for factor in adjustmentFactor:
            # Adjust the basevalue in the optimizer
            optAdjusted = deepcopy(optimizerBaseCase)
            setattr(optAdjusted, hyperparamName, baseval * factor)

            # Save the resulting optimizer
            optVariants[hyperparamName].append(optAdjusted)

    # Run optimization/training, with all variations of each hyperparameter "in parallel", e.g. "lr" = 0.1, 0.2, 0.3 with the same random batches
    for hyperparamName, optimizerList in optVariants.items():
        # NOTE: There might be an issue where each optimizer have a deepcopy of their respective lossObj, meaning they don't use the same batching
        train(optimizerList, nrEpochs)  

    return optVariants

def main():
    """ 
    Test by passing each optimizer to testHyperparameterSensitivity.
    NOTE: Issues with mapping the total optimizer hyperparameter settings to each result, as of now.
    """
    A = np.array([
        [19, 0],
        [0, 5]
        ])
    b = np.array([1, 5])
    lossObj = QuadraticForm(A, b)
    initPos = np.array([1.0, 1.0])

    # Setup of optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr=0.5)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.5, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.5, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.5, forgettingFactorM=0.9, forgettingFactorR=0.999)

    optimizerList = [optSGD, optNesterov, optMomentum, optAdam]
    for opt in optimizerList:
        optVariants = testHyperparameterSensitivity(opt, nrEpochs=5) # the test will generate variants of the optimizer and train them

        print(f" \n Optimizer: {opt.__class__.__name__}")
        for hyperparamName, optimizerList in optVariants.items():
            print(f"\t Variation in {hyperparamName}")
            plt.figure()
            for opt in optimizerList:
                lossHistory = opt.lossHistory
                print(f"\t Loss history for {opt.getHyperparamDict()}: {hyperparamName}")

                # Plot all loss history in one graph, alt. use: plt.subplot(2,2, i+1), add index i
                plotHistoryGraph(lossHistory, f"{opt.__class__.__name__} Loss History \n {opt.getHyperparamDict}", ylabel="Loss")
            plt.show()

def mainAlt():
    A = np.array([
        [19, 0],
        [0, 5]
        ])
    b = np.array([1, 5])
    lossObj = QuadraticForm(A, b)
    initPos = np.array([1.0, 1.0])

    # SGD
    optSGD = sgd.SGD(lossObj, initPos, lr=0.1)
    optGroupSGD = optimizerGroup(optimizerBaseCase = optSGD)

    # Momentum
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.3, decayFactor=0.9)
    optGroupMomentum = optimizerGroup(optimizerBaseCase = optSGD)

    # Nesterov
    optAdam = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    optGroupNesterov = optimizerGroup(optimizerBaseCase = optSGD)

    # Adam
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)
    optGroupAdam = optimizerGroup(optimizerBaseCase = optSGD)

    optimizerGroupList = [optGroupSGD, optGroupMomentum, optGroupNesterov, optGroupAdam]
    for optGroup in optimizerGroupList:
        # Train all variants. For example, all variations of "lr" = 0.01, 0.1, 1 will be trained with the same random batches. 
        optGroup.trainAllVariants()

        # Present
        # TODO

if __name__ == "__main__": 
    main()
