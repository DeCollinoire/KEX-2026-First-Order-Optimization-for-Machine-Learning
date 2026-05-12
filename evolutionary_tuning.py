import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

# Project files
from utils import plotHistoryGraph, train, train_external_batching, setupProblem
from DataLoader import loadDataAsNumpyArray
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock
from optimizers import sgd, nesterov, momentum, adam
import mplcursors
import random

"""
NOTE: This version implements a gridsearch approach to sensitivity, i.e. it tests all combinations of the given hyperparams.
All optimizer variants are created first, as opposed to modifying them one-by-one during testing.
"""

class LogisticRegression_ExternalBatching():
    """ 
    This version of logistic regression externalizes batching from the lossObj, which allows us to use the same lossObj for all optimizers and batch sizes in the sensitivity test.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def setCurrentBatch(self, X_batch, y_batch):
        self.X_batch = X_batch
        self.y_batch = y_batch

    def getCurrentBatch(self):
        return self.X_batch, self.y_batch

    def evaluate_loss(self, weights):
        nrOfSamples = self.X.shape[0]
        return np.clip(np.sum(np.log(1 + np.exp(-self.y * (self.X @ weights)))) / nrOfSamples, -1e6, 1e6)

    def evaluate_gradient(self, weights):
        X, y = self.getCurrentBatch()
        return -(X.T @ (y * expit(-(y * (X @ weights))))) # expit is a sigmoid function

def cost(optimizer):
    """ 
    We want to penalize:
        - High final loss
        - High loss after the first epoch (approximates slow convergence)
        - High variance in the loss history (i.e. oscillations)
    """
    cost = 0
    cost += optimizer.lossHistory[-1] # final loss
    cost += 2*optimizer.lossHistory[0] # loss after first epoch
    cost += 2*optimizer.lossHistory[1] # loss after second epoch
    cost += 0.5*np.var(optimizer.lossHistory) # variance in loss history
    
    # hyperparamCost = sum([abs(value) for value in optimizer.getHyperparamDict().values()])
    # cost += 0.1 * hyperparamCost # Add a small cost for large hyperparameter values to encourage reasonable hyperparameters
    
    if cost is np.inf or np.isnan(cost):
        cost = 1e6 # Assign a large cost for infinite or NaN values
    return cost

def createVariants(lossObj, initPos, populationSize=10):
    variants = {"SGD": [], "Nesterov": [], "Momentum": [], "Adam": []}

    # SGD
    for i in range(populationSize):
        variants["SGD"].append(sgd.SGD(lossObj, initPos, lr=random.uniform(0.001, 1)))
        variants["Nesterov"].append(nesterov.Nesterov(lossObj, initPos, lr=random.uniform(0.001, 1), decayFactor=random.uniform(0.1, 0.999)))
        variants["Momentum"].append(momentum.Momentum(lossObj, initPos, learningRate=random.uniform(0.001, 1), decayFactor=random.uniform(0.1, 0.999)))
        variants["Adam"].append(adam.Adam(lossObj, initPos, learningRate=random.uniform(0.001, 1), forgettingFactorM=random.uniform(0.1, 0.99), forgettingFactorR=random.uniform(0.1, 0.99)))
    return variants

def mutateOptimizer(opt):
    optClass = opt.__class__
    hyperparams = opt.getHyperparamDict().copy()
    for hyperparamName, hyperparamValue in hyperparams.items():
        # Randomly perturb the hyperparameter by multiplying it with a random factor
        randomFactor = np.random.uniform(0.5, 1.2)
        hyperparams[hyperparamName] = hyperparamValue * randomFactor
    return optClass(opt.lossObj, opt.initPos, **hyperparams)

def evolutionaryTuning(optimizerList, lossObj, X, y, nrEpochs=20, batchSize=1000, n_generations=5):
    """
    This test implements an evolutionary approach.
    1. Train the optimizers in optimizerList for nrEpochs.
    2. Evaluate the cost of each optimizer.
    3. Select the top 50% optimizers and mutate their hyperparameters.
    """
    for gen in range(n_generations):
        print(f"Generation {gen+1}/{n_generations}")
        # Reset loss history and position
        for opt in optimizerList:
            opt.reset()

        # Train optimizers
        train_external_batching(optimizerList, lossObj, X, y, nrEpochs=nrEpochs, batchSize=batchSize)

        # Evaluate cost
        costs = [cost(opt) for opt in optimizerList]

        # Select top 50%
        sortedIndices = np.argsort(costs)
        topOptimizers = [optimizerList[i] for i in sortedIndices[:len(optimizerList)//2]]

        # Mutate hyperparameters of top optimizers and create new optimizer list
        optimizerList = topOptimizers.copy() # Start with the top optimizers (unmutated)
        for opt in topOptimizers:
            # Create a new mutated 
            mutatedOpt = mutateOptimizer(opt)
            optimizerList.append(mutatedOpt)
        
    # Return the final list of optimizers sorted by cost
    train_external_batching(optimizerList, lossObj, X, y, nrEpochs=nrEpochs, batchSize=batchSize) # Final training to update loss history for plotting
    costs = [cost(opt) for opt in optimizerList]
    sortedIndices = np.argsort(costs)
    optimizerList = [optimizerList[i] for i in sortedIndices[:len(optimizerList)//2]]
    return optimizerList

datasetMap = {
    "australian_scale": "datasets/australian_scale",
}

def main():
    # Setup problem
    problemName = "australian_scale"
    datasetFilepath = datasetMap[problemName]
    X, y = loadDataAsNumpyArray(datasetFilepath)
    lossObj = LogisticRegression_ExternalBatching(X, y)
    initPos = np.zeros(X.shape[1]) # type: ignore
    variantsDict = createVariants(lossObj, initPos, populationSize=10)

    # Run evolutionary tuning
    tunedOptimizerDict = {}
    for optVariants in variantsDict.values():
        tunedOptimizerDict[optVariants[0].__class__.__name__] = evolutionaryTuning(optVariants, lossObj, X, y, n_generations=30)

    # Plot
    for optimizerList in tunedOptimizerDict.values():
        plt.figure(str(optimizerList[0].__class__.__name__+"_tuning"), figsize=(10, 6))
        for opt in optimizerList:
            plotHistoryGraph(opt.lossHistory, title=f"{opt.__class__.__name__} Hyperparameter Sensitivity, lossObj = {lossObj.__class__.__name__}, problem = {datasetFilepath}", label=f"{opt.__class__.__name__}, {opt.getHyperparamStr()}", ylabel="Loss", yscale="linear")
        plt.grid()

        # Adding interactability
        cursor = mplcursors.cursor(hover=False)

        @cursor.connect("add")
        def _(sel):
            sel.annotation.set_text(sel.artist.get_label())
            sel.artist.set_linewidth(9)

        @cursor.connect("remove")
        def _(sel):
            # Reset linewidth when clicking away
            sel.artist.set_linewidth(1.5)
        plt.show()

if __name__ == "__main__":
    main()
