import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
import mplcursors

# Project files
from utils import plotHistoryGraph, train, train_external_batching, setupProblem
from DataLoader import loadDataAsNumpyArray
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock
from optimizers import sgd, nesterov, momentum, adam

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
        return np.sum(np.log(1 + np.exp(-self.y * (self.X @ weights))))

    def evaluate_gradient(self, weights):
        X, y = self.getCurrentBatch()
        return -(X.T @ (y * expit(-(y * (X @ weights))))) # expit is a sigmoid function

def createVariants(lossObj, initPos, rcv1_on = True):
    variants = {"SGD": [], "Nesterov": [], "Momentum": [], "Adam": []}

    # SGD
    lrList = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
    for lr in lrList:
        variants["SGD"].append(sgd.SGD(lossObj, initPos, lr=lr))

    # Nesterov
    if rcv1_on:
        lrList = [0.01, 0.025, 0.05, 0.075, 0.1]
        decayFactorList = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99]
    else:
        lrList = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
        decayFactorList = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.999]
    for lr in lrList:
        for decayFactor in decayFactorList:
            variants["Nesterov"].append(nesterov.Nesterov(lossObj, initPos, lr=lr, decayFactor=decayFactor))
    
    # Momentum
    # Use same settings as Nesterov
    for lr in lrList:
        for decayFactor in decayFactorList:
            variants["Momentum"].append(momentum.Momentum(lossObj, initPos, learningRate=lr, decayFactor=decayFactor))

    # Adam
    if rcv1_on:
        lrList = [0.1, 0.5, 0.85, 0.1, 0.5, 1]
        forgettingFactorMList = [0.7, 0.8, 0.9, 0.99]
        forgettingFactorRList = [0.7, 0.8, 0.9, 0.99]
    else:
        lrList = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        forgettingFactorMList = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        forgettingFactorRList = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    for lr in lrList:
        for forgettingFactorM in forgettingFactorMList:
            for forgettingFactorR in forgettingFactorRList:
                variants["Adam"].append(adam.Adam(lossObj, initPos, learningRate=lr, forgettingFactorM=forgettingFactorM, forgettingFactorR=forgettingFactorR))

    return variants

def main():
    # Config
    randomSeed = 10
    datasetFilepath = "datasets/australian_scale" # This is also used for plot titles
    initialPosInterval = 0
    batchSize = 100000
    l2NormalizationOn = (datasetFilepath == "datasets/australian") or (datasetFilepath == "datasets/australian_scale")

    # Setup
    print("Setting up...")
    np.random.seed(randomSeed)
    X, y = loadDataAsNumpyArray(datasetFilepath, toDense=False, l2NormalizationOn=l2NormalizationOn)
    nrSamples, nrFeatures = X.shape # type: ignore

    lossObj = LogisticRegression_ExternalBatching(X, y) # Use the external batching version of logistic regression for this test. It works fine to use the internal batching of the lossObj also.
    initPos = np.random.uniform(-initialPosInterval, initialPosInterval, nrFeatures) # Initialize position based on the number of features in the dataset

    # Create variants
    variants = createVariants(lossObj, initPos, rcv1_on = not l2NormalizationOn) # Creates all variants of sgd, nesterov, momentum and adam in a dictionary
    joinedOptList = [opt for optList in variants.values() for opt in optList] # Join all optimizers into one list for training
    print(f"Setup finished!")

    # Train
    print(f"\n--- Training ---")
    train_external_batching(joinedOptList, lossObj, X, y, nrEpochs=20, batchSize=batchSize, printProgress=True)
    print(f"Training finished.")

    # Plot
    for optClassName, optVariants in variants.items():
        plt.figure(str(optClassName+"_sensitivity_test"), figsize=(10, 6))
        lines = []
        
        for opt in optVariants:
            # Pass the hyperparameter string into the label or a custom attribute
            label_str = f"{opt.__class__.__name__}: {opt.getHyperparamStr()}"
            line = plotHistoryGraph(opt.lossHistory, 
                                    title=f"{optClassName} Sensitivity, dataset = {datasetFilepath}", 
                                    label=label_str, 
                                    ylabel="Loss",
                                    legendOn=False # use external legend
                                    )
            lines.append(line)
        # Position legend outside to the right
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout() # Adjust layout to make room for the legend
        
        # Adding interactability
        cursor = mplcursors.cursor(hover=False)

        @cursor.connect("add")
        def _(sel):
            sel.annotation.set_text(sel.artist.get_label())
            sel.artist.set_linewidth(6)

        @cursor.connect("remove")
        def _(sel):
            # Reset linewidth when clicking away
            sel.artist.set_linewidth(1.5)
    plt.show()

if __name__ == "__main__":
    main()
