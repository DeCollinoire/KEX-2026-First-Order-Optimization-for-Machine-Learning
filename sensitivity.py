import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

# Project files
from utils import plotHistoryGraph, train, train_external_batching, setupProblem
from DataLoader import loadDataAsNumpyArray
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock
from optimizers import sgd, nesterov, momentum, adam


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
        return np.sum(np.log(1 + np.exp(-self.y * (self.X @ weights)))) / nrOfSamples

    def evaluate_gradient(self, weights):
        X, y = self.getCurrentBatch()
        return -(X.T @ (y * expit(-(y * (X @ weights))))) # expit is a sigmoid function

def createVariants(lossObj, initPos):
    variants = {"SGD": [], "Nesterov": [], "Momentum": [], "Adam": []}

    # SGD
    for lr in [0.01, 0.1, 0.2, 0.5, 1, 5, 10, 100, 1000, 2000]:
        variants["SGD"].append(sgd.SGD(lossObj, initPos, lr=lr))

    # Nesterov
    for lr in [0.01, 0.1, 0.2, 0.5, 1, 5, 10]:
        for decayFactor in [0.5, 0.7, 0.9, 0.99]:
            variants["Nesterov"].append(nesterov.Nesterov(lossObj, initPos, lr=lr, decayFactor=decayFactor))
    
    # Momentum
    for lr in [0.01, 0.1, 0.2, 0.5, 1, 5, 10]:
        for decayFactor in [0.5, 0.7, 0.9, 0.99]:
            variants["Momentum"].append(momentum.Momentum(lossObj, initPos, learningRate=lr, decayFactor=decayFactor))

    # Adam
    for lr in [0.01, 0.1, 0.2, 0.5, 0.75]:
        for forgettingFactorM in [0.7, 0.9, 0.99]:
            for forgettingFactorR in [0.7, 0.9, 0.99]:
                variants["Adam"].append(adam.Adam(lossObj, initPos, learningRate=lr, forgettingFactorM=forgettingFactorM, forgettingFactorR=forgettingFactorR))

    return variants

def main():
    # Config
    randomSeed = 25
    datasetFilepath = "datasets/rcv1_train.binary" # This is also used for plot titles
    initialPosInterval = 0.1
    batchSize = 1000

    # Setup
    print("Setting up...")
    np.random.seed(randomSeed)
    X, y = loadDataAsNumpyArray(datasetFilepath, toDense=False, L2NormalizationOn=False)
    nrSamples, nrFeatures = X.shape # type: ignore

    lossObj = LogisticRegression_ExternalBatching(X, y) # Use the external batching version of logistic regression for this test
    initPos = np.random.uniform(-initialPosInterval, initialPosInterval, nrFeatures) # Initialize position based on the number of features in the dataset

    # Create variants
    variants = createVariants(lossObj, initPos) # Creates all variants of sgd, nesterov, momentum and adam in a dictionary
    joinedOptList = [opt for optList in variants.values() for opt in optList] # Join all optimizers into one list for training
    print(f"Setup finished!")

    # Train
    print(f"\n--- Training ---")
    train_external_batching(joinedOptList, lossObj, X, y, nrEpochs=20, batchSize=batchSize, printProgress=True)
    print(f"Training finished.")

    # Plot
    for optClassName, optVariants in variants.items():
        plt.figure(str(optClassName+"_sensitivity_test"), figsize=(10, 6))
        for opt in optVariants:
            plotHistoryGraph(opt.lossHistory, title=f"{opt.__class__.__name__} Hyperparameter Sensitivity, lossObj = {lossObj.__class__.__name__}, problem = {datasetFilepath}", label=f"{opt.__class__.__name__}, {opt.getHyperparamStr()}", ylabel="Loss", yscale="linear")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
