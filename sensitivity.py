import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
import mplcursors
import itertools

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
    This version of logistic regression externalizes batching from the lossObj.
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


def createVariants(lossObj, initPos, problemName="australian_scale"):
    # Master configuration for each problem
    configs = {
        "australian_scale": {
            "SGD": {"lr": [0.0005, 0.0006, 0.0007, 0.0008, 0.001, 0.002, 0.003, 0.004, 0.005]},
            "Nesterov": {
                "lr": [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
                "df": [0.5, 0.6, 0.7, 0.9, 0.99, 0.999]
            },
            "Adam": {
                "lr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "fm": [0.7, 0.8, 0.9, 0.99],
                "fr": [0.7, 0.8, 0.9, 0.99]
            }
        },
        "rcv1": {
            "SGD": {"lr": [0.01, 0.025, 0.05, 0.075, 0.1]},
            "Nesterov": {
                "lr": [0.01, 0.025, 0.05, 0.075, 0.1],
                "df": [0.5, 0.8, 0.95, 0.999]
            },
            "Adam": {
                "lr": [0.01, 0.1, 0.5],
                "fm": [0.4, 0.7, 0.9],
                "fr": [0.4, 0.7, 0.9]
            }
        },
        "Rosenbrock": {
            "SGD": {
                "lr": [0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009],
            },
            "Nesterov": {
                "lr": [0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009],
                "df": [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
            },
            "Adam": {
                "lr": [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
                "fm": [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999],
                "fr": [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
            }
        }
    }

    # Extract the specific config for the current problem
    problemConfig = configs.get(problemName, configs["australian_scale"]) # default to australian
    variants = {opt: [] for opt in ["SGD", "Nesterov", "Momentum", "Adam"]}

    # SGD
    for lr in problemConfig["SGD"]["lr"]:
        variants["SGD"].append(sgd.SGD(lossObj, initPos, lr=lr))

    # Nesterov & Momentum (sharing Nesterov config)
    nesterovConfig = problemConfig["Nesterov"]
    for lr, df in itertools.product(nesterovConfig["lr"], nesterovConfig["df"]):
        variants["Nesterov"].append(nesterov.Nesterov(lossObj, initPos, lr=lr, decayFactor=df))
        variants["Momentum"].append(momentum.Momentum(lossObj, initPos, learningRate=lr, decayFactor=df))

    # Adam
    adamConfig = problemConfig["Adam"]
    for lr, fm, fr in itertools.product(adamConfig["lr"], adamConfig["fm"], adamConfig["fr"]):
        variants["Adam"].append(adam.Adam(lossObj, initPos, learningRate=lr, forgettingFactorM=fm, forgettingFactorR=fr))

    return variants


datasetMap = {
    "Rosenbrock": "N/A",
    "australian": "datasets/australian",
    "australian_scale": "datasets/australian_scale",
    "rcv1": "datasets/rcv1_train.binary"
}

def main():
    # Config
    randomSeed = 25
    problemName = "Rosenbrock"
    datasetFilepath = datasetMap[problemName] # This is also used for plot titles
    batchSize = 100000 # None means fullbatch, only used by LogReg
    initialPosInterval = 0
    l2NormalizationOn = (problemName == "australian") or (problemName == "australian_scale")
    dim = 2 # For Rosenbrock only
    nrEpochs = 100
    printProgress=True

    # Setup
    print("Setting up...")
    np.random.seed(randomSeed)
    lossObj, initPos = setupProblem(problemName=problemName, dim=dim, datasetFilepath=datasetFilepath,
                                    initialPosInterval=initialPosInterval, randomSeed=randomSeed,
                                    batchSize=batchSize, toDense=False, l2NormalizationOn=l2NormalizationOn)  # QDF, Rosenbrock; datasetFilepath is only needed for LogReg

    # Create variants
    variants = createVariants(lossObj, initPos, problemName=problemName) # Creates all variants of sgd, nesterov, momentum and adam in a dictionary
    joinedOptList = [opt for optList in variants.values() for opt in optList] # Join all optimizers into one list for training
    print(f"Setup finished!")

    # Train
    print(f"\n--- Training ---")
    # train_external_batching(joinedOptList, lossObj, X, y, nrEpochs=nrEpochs, batchSize=batchSize, printProgress=printProgress)
    train(joinedOptList, lossObj, nrEpochs=nrEpochs, printProgress=printProgress)
    print(f"Training finished.")

    # Plot
    for optClassName, optVariants in variants.items():
        plt.figure(str(optClassName+"_sensitivity_test"), figsize=(10, 6))
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.8, top=None, wspace=None, hspace=0.9)
        lines = []
        
        for opt in optVariants:
            # Pass the hyperparameter string into the label or a custom attribute
            label_str = f"{opt.__class__.__name__}: {opt.getHyperparamStr()}"
            line = plotHistoryGraph(opt.lossHistory, 
                                    title=f"{optClassName} Sensitivity, {problemName}, dim = {lossObj.xDataLength}, dataset = {datasetFilepath}, batchSize = {"fullbatch" if batchSize is None else batchSize}", 
                                    label=label_str, 
                                    ylabel="Loss",
                                    legendOn=False # use external legend
                                    )
            lines.append(line)
        # Position legend outside to the right
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        # plt.tight_layout() # Adjust layout to make room for the legend
        
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
