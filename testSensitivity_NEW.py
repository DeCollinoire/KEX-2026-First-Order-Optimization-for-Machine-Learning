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

def linspace(start, end, nrPoints=10, delta=None):
    """ Generate nrPoints evenly separated points between start and end. """
    if delta is None:
        delta = abs(end - start) / (nrPoints - 1)
    else:
        # If delta is provided, calculate the number of points needed to cover the range from start to end with that delta
        nrPoints = int(abs(end - start) / delta) + 1
    return [float(f"{(start + delta * i):.3g}") for i in range(nrPoints)]     # Use f-string with to get a certain number of significant figures, then convert back to float. nrPoints to include the end point

def linspace_centered(center, variance, nrPoints, onlyPositive = True):
    """ Generate nrPoints evenly separated points between center-variance and center+variance. """
    start = center - variance
    if onlyPositive and center - variance < 0:
        start = 0.0001
    return linspace(start, center + variance, nrPoints)

def exp_space(start, end, nrPoints):
    """ Generate nrPoints exponentially separated points between start and end. """
    return [float(f"{(start * (end/start) ** (i/(nrPoints-1))):.3g}") for i in range(nrPoints)]     # Use f-string with to get a certain number of significant figures, then convert back to float. nrPoints to include the end point

def binary_space(start, end, nrPoints):
    """ Generate nrPoints binary separated points between start and end. """
    return [float(f"{(start * (2 ** i)):.3g}") for i in range(nrPoints) if start * (2 ** i) <= end]     # Use f-string with to get a certain number of significant figures, then convert back to float. nrPoints to include the end point

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
    results = deepcopy(baseCase.getHyperparamDict())
    hyperparamsBaseCase = baseCase.getHyperparamDict()

    # For each hyperparameter, generate adjusted optimizers (only downward adjustments to prevent too large values)
    for hyperparamName, basevalue in hyperparamsBaseCase.items():
        # Initialize result list
        results[hyperparamName] = [] # [baseCase] # Include base case for comparison
        testValues = hyperparamConfig[hyperparamName]
        
        # Adjust the hyperparameter values
        for value in testValues:
            hyperparamsTEMP = deepcopy(hyperparamsBaseCase) # Alternatively, create an empty dictionary and insert all values of the base case
            hyperparamsTEMP[hyperparamName] = value

            # Create and save the adjusted optimizer
            opt = optimizerBaseCaseClass(baseCase.lossObj, baseCase.pos.copy(), **hyperparamsTEMP) # Creates a new optimizer
            results[hyperparamName].append(opt)
            
        # Run the optimization/training for all optimizers with the 'hyperparamName' parameter adjusted. Histories are stored in the optimizer objects
        train(optimizerList=results[hyperparamName], nrEpochs=nrEpochs)

    return results

optimizerBaseCaseConfig = {
    "Rosenbrock": {
        "SGD": {"lr": 0.001},
        "Momentum": {"learningRate": 0.001, "decayFactor": 0.9},
        "Nesterov": {"lr": 0.001, "decayFactor": 0.9},
        "Adam": {"learningRate": 0.01, "forgettingFactorM": 0.9, "forgettingFactorR": 0.999}
    },
    "australian": { # these are assumed to be the same as for australian_scale
        "SGD": {"lr": 0.04},
        "Momentum": {"learningRate": 0.025, "decayFactor": 0.9},
        "Nesterov": {"lr": 0.0035, "decayFactor": 0.9},
        "Adam": {"learningRate": 0.1, "forgettingFactorM": 0.9, "forgettingFactorR": 0.999}
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

datasetMap = {
    "australian": "datasets/australian",
    "australian_scale": "datasets/australian_scale",
    "rcv1": "datasets/rcv1_train.binary"
}

def main():
    # Config
    problemName = "australian_scale"
    datasetFilepath = datasetMap.get(problemName, "N/A")
    randomSeed = 25
    initialPosInterval = 0
    nrEpochs = 16
    dim = 10 # Used by Rosenbrock for initPos

    # LogReg-Specific Config
    batchSize = 64 # Full batch: 20242+ for rcv1, 690+ for australian
    toDense = False
    l2NormalizationOn = (problemName in ["australian", "australian_scale"])
    
    # Setup loss object
    print("Setting up...")
    lossObj, initPos = setupProblem(problemName, datasetFilepath = datasetFilepath, randomSeed = randomSeed, initialPosInterval = initialPosInterval, batchSize = batchSize, toDense = toDense, l2NormalizationOn = l2NormalizationOn, dim = dim)
    print(f"Data loaded and lossObj created")

    # Setup base case optimizers
    optSGD = sgd.SGD(lossObj, initPos, **optimizerBaseCaseConfig[problemName]["SGD"])
    optNesterov = nesterov.Nesterov(lossObj, initPos, **optimizerBaseCaseConfig[problemName]["Nesterov"])
    optMomentum = momentum.Momentum(lossObj, initPos, **optimizerBaseCaseConfig[problemName]["Momentum"])
    optAdam = adam.Adam(lossObj, initPos, **optimizerBaseCaseConfig[problemName]["Adam"])

    # Define an associated hyperparameter config dictionary if desired (missing values will be generated in the test)
    optimizerList = [
        (optSGD, {"lr": binary_space(0.01, 0.2, 10)}),
        (optNesterov, {"lr": linspace(0.01, 0.5, 10), "decayFactor": linspace(0.3, 0.99, 10)}),
        (optMomentum, {"learningRate": linspace(0.01, 0.5, 10), "decayFactor": linspace(0.3, 0.99, 10)}),
        (optAdam, {"learningRate": linspace(0.05, 1.75, 10), "forgettingFactorM": linspace(0.3, 0.99, 10), "forgettingFactorR": [0.05, 0.1]+linspace(0.5, 0.999, 7)+[0.9999]})
        ]
    print("Optimizers created")

    # Run the test for each optimizer
    for opt, hConfig in optimizerList:
        print(f"\n --- Testing {opt.__class__.__name__} --- ")
        results = test_hyperparameter_sensitivity(opt, hyperparamConfig=hConfig, nrEpochs = nrEpochs)

        # Present
        print("Testing finished!")

        # Plotting
        i = 0
        plt.figure(str(opt.__class__.__name__+"_sensitivity_test"), figsize=(10, 9))
        for hyperparamName, optimizerListCopy in results.items():
            # Plot all variations in the same graph, but each hyperparameter gets one subplot.
            nrOfHyperparams = len(optimizerListCopy[0].getHyperparamDict().keys())
            plt.subplot(nrOfHyperparams, 1, i+1); plt.subplots_adjust(right=0.8, hspace=0.3) 
            for optVariation in optimizerListCopy:
                plotHistoryGraph(optVariation.lossHistory, 
                                 title = f"Loss for {optVariation.__class__.__name__}, variation in '{hyperparamName}', lossObj = {optVariation.lossObj.__class__.__name__}, batchSize = {optVariation.lossObj.batchSize}, dataset = {datasetFilepath}, randomSeed = {randomSeed}", 
                                 label=f"{optVariation.__class__.__name__}, {optVariation.getHyperparamStr()}", 
                                 ylabel="Loss",
                                 marker="o", markersize=4, linewidth=2)
                plt.minorticks_on()
                plt.grid(True, which="minor", linestyle=":", linewidth=1)

                # Put legends outside of the plot area and increase line marker size
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            i += 1
            plt.savefig("images/"+opt.__class__.__name__+"_sensitivity_test.png", dpi=300, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__": 
    main()