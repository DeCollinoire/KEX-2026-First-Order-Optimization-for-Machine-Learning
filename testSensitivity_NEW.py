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

def test_hyperparameter_sensitivity(baseCase: Optimizer, hyperparamConfig = dict(), nrEpochs = 100):
    """"
    Results will be a dictionary on the following format:
        results = {
            "h1": [opt11, opt12] # hyperparam1
            "h1": [opt21, opt22] # hyperparam2
        }
    where opt11 is the basecase with the first hyperparameter set to the first variation of that hyperparameter.
    Use hyperparameterConfig to manually set predefined values for the tests, as opposed to generating from base value.
    """
    # Setup
    optimizerBaseCaseClass = baseCase.__class__
    lossObj = baseCase.lossObj
    initPos = baseCase.pos.copy()
    results = deepcopy(baseCase.getHyperparamDict())
    hyperparamsBaseCase = baseCase.getHyperparamDict()

    # For each hyperparameter, generate adjusted optimizers (only downward adjustments to prevent too large values)
    adjustmentFactors = [0.001, 0.01, 0.1, 1]
    for hyperparamName, basevalue in hyperparamsBaseCase.items():
        # Initialize result list
        results[hyperparamName] = []

        # If values are not manually set in 'hyperparameterConfig', then generate test values from 'basevalue'
        if hyperparamConfig.get(hyperparamName, None) is None: 
            testValues = [factor * basevalue for factor in adjustmentFactors]
        else:
            testValues = hyperparamConfig[hyperparamName]
        
        # Adjust the hyperparameter values
        for value in testValues:
            hyperparamsTEMP =  deepcopy(hyperparamsBaseCase) # Alternatively, create an empty dictionary and insert all values of the base case
            hyperparamsTEMP[hyperparamName] = value

            # Create and save the adjusted optimizer
            opt = optimizerBaseCaseClass(lossObj, initPos, **hyperparamsTEMP)
            results[hyperparamName].append(opt)
            
        # Run the optimization/training for all optimizers with the 'hyperparamName' parameter adjusted. Histories are stored in the optimizer objects
        train(optimizerList=results[hyperparamName], nrEpochs=nrEpochs)

    return results

def main():
    # Setup loss object
    lossObj, initPos = setupProblem("LogReg", datasetFilepath="rcv1_train.binary")  # australian_scaled, australian, rcv1_train.binary
    #lossObj, initPos = setupProblem("Rosenbrock", dim=10)
    #lossObj, initPos = setupProblem("QDF")

    # Setup base case optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr=0.1)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)

    # Define an associated hyperparameter config dictionary if desired (missing values will be generated in the test)
    optimizerList = [
        (optSGD, {"lr": [0.001, 0.01, 0.1, 0.5, 1, 2]}),
        (optNesterov, dict()),
        (optMomentum, dict()),
        (optAdam, {"learningRate": [0.01, 0.1, 0.78, 1]})
        ]

    # Run the test for each optimizer
    for opt, hConfig in optimizerList:
        print(f" --- {opt.__class__.__name__} --- ")
        results = test_hyperparameter_sensitivity(opt, hyperparamConfig=hConfig, nrEpochs = 50)

        # Present
        # TODO: Print quantitative measurements
        print("...")

        # Plotting
        i = 0
        plt.figure(str(opt.__class__.__name__+"_sensitivity_test"), figsize=(10, 6))
        for hyperparamName, optimizerList in results.items():
            # Plot all variations in the same graph, but each hyperparameter gets one subplot.
            plt.subplot(3, 1, i+1)
            plt.subplots_adjust(right=0.8)
            for optVariation in optimizerList:
                plotHistoryGraph(optVariation.lossHistory, title = f"Loss for {optVariation.__class__.__name__}, variation in '{hyperparamName}'", label=f"{optVariation.__class__.__name__}, {optVariation.getHyperparamStr()}", ylabel="Loss")
            i += 1
        plt.savefig("images/"+opt.__class__.__name__+"_sensitivity_test.png", dpi=300, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__": 
    main()