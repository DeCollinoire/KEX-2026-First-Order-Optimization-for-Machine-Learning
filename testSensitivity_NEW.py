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
    # Config
    problemName = "Rosenbrock"
    datasetFilepath = "datasets/rcv1_train.binary" # australian_scale, australian, rcv1_train.binary # This is also used to show the name in the plot title
    randomSeed = 25
    initialPosInterval = 0.1
    dim = 10 # Used by Rosenbrock for initPos

    # LogReg-Specific Config
    batchSize = 1000000 # Full batch: 20242+ for rcv1, 690+ for australian
    toDense = False
    l2NormalizationOn = False
    
    # Setup loss object
    print("Setting up...")
    lossObj, initPos = setupProblem(problemName, datasetFilepath = datasetFilepath, randomSeed = randomSeed, initialPosInterval = initialPosInterval, batchSize = batchSize, toDense = toDense, l2NormalizationOn = l2NormalizationOn, dim = dim)
    print(f"Data loaded and lossObj created")

    # Setup base case optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr=0.07)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.07, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.07, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)

    # Define an associated hyperparameter config dictionary if desired (missing values will be generated in the test)
    optimizerList = [
        (optSGD, {"lr": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.002, 0.003, 0.004]}),
        (optNesterov, {"lr": [0.01, 0.05, 0.07, 0.075, 0.08, 0.1], "decayFactor": [0.3, 0.5, 0.7, 0.8, 0.85, 0.9]}),
        (optMomentum, {"learningRate": [0.01, 0.05, 0.07, 0.075, 0.08, 0.1], "decayFactor": [0.5, 0.7, 0.9, 0.99]}),
        (optAdam, {"learningRate": [0.01, 0.1, 0.2, 0.25, 0.5, 1, 2, 3], "forgettingFactorM": [0.5, 0.7, 0.9, 0.99], "forgettingFactorR": [0.5, 0.7, 0.9, 0.99]})
        ]
    print("Optimizers created")

    # Run the test for each optimizer
    for opt, hConfig in optimizerList:
        print(f"\n --- Testing {opt.__class__.__name__} --- ")
        results = test_hyperparameter_sensitivity(opt, hyperparamConfig=hConfig, nrEpochs = 50)

        # Present
        # TODO: Print quantitative measurements
        print("Testing finished!")

        # Plotting
        i = 0
        plt.figure(str(opt.__class__.__name__+"_sensitivity_test"), figsize=(6, 6))
        for hyperparamName, optimizerListCopy in results.items():
            # Plot all variations in the same graph, but each hyperparameter gets one subplot.
            nrOfHyperparams = len(optimizerListCopy[0].getHyperparamDict().keys())
            plt.subplot(nrOfHyperparams, 1, i+1); plt.subplots_adjust(right=0.8)
            for optVariation in optimizerListCopy:
                plotHistoryGraph(optVariation.lossHistory, title = f"Loss for {optVariation.__class__.__name__}, variation in '{hyperparamName}', lossObj = {optVariation.lossObj.__class__.__name__}, batchSize = {optVariation.lossObj.batchSize}, dataset = {datasetFilepath}, randomSeed = {randomSeed}", label=f"{optVariation.__class__.__name__}, {optVariation.getHyperparamStr()}", ylabel="Loss")
            i += 1
            plt.savefig("images/"+opt.__class__.__name__+"_sensitivity_test.png", dpi=300, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__": 
    main()