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

def linspace(start, end, nrPoints):
    """ Generate nrPoints evenly separated points between start and end. """
    delta = abs(end - start) / (nrPoints - 1)
    return [float(f"{(start + delta * i):.3g}") for i in range(nrPoints)]     # Use f-string with :.3g for 3 significant figures, then convert back to float. nrPoints to include the end point

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
            "SGD": {"lr": linspace(0.001, 0.1, 50)},
            "Nesterov": {
                "lr": linspace(0.001, 0.01, 20),
                "df": linspace(0.5, 0.999, 10)
            },
            "Adam": {
                "lr": linspace(0.1, 0.6, 10),
                "fm": linspace(0.7, 0.99, 5),
                "fr": linspace(0.7, 0.999, 5)
            }
        },
        "rcv1": {
            "SGD": {"lr": linspace(0.01, 0.1, 10)},
            "Nesterov": {
                "lr": linspace(0.001, 0.1, 20),
                "df": linspace(0.5, 0.999, 10)
            },
            "Adam": {
                "lr": linspace(0.01, 0.1, 10),
                "fm": linspace(0.6, 0.99, 5),
                "fr": linspace(0.6, 0.999, 5)
            }
        },
        "Rosenbrock": {
            "SGD": {
                "lr": linspace(0.001, 0.002, 25),
            },
            "Nesterov": {
                "lr": linspace(0.00001, 0.01, 25),
                "df": linspace(0.8, 0.95, 15)
            },
            "Adam": {
                "lr": linspace(0.002, 0.02, nrPoints=6),
                "fm": linspace(0.8, 0.99, nrPoints=4), # [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999],
                "fr": linspace(0.8, 0.999, nrPoints=4) 
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
    problemName = "australian_scale"
    datasetFilepath = datasetMap[problemName]
    batchSize = 100000 # Use large batch size to get full batch.
    initialPosInterval = 0
    l2NormalizationOn = (problemName in ["australian", "australian_scale"])
    dim = 10 # For Rosenbrock only
    nrEpochs = 250
    printProgress = True

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
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=None, wspace=None, hspace=0.9)
        lines = []
        
        for opt in optVariants:
            # Pass the hyperparameter string into the label or a custom attribute
            label_str = f"{opt.__class__.__name__}: {opt.getHyperparamStr()}"
            line = plotHistoryGraph(opt.lossHistory, 
                                    title=f"{optClassName} Sensitivity, {problemName}, dim = {lossObj.xDataLength}, dataset = {datasetFilepath}, batchSize = {"fullbatch" if batchSize is None else batchSize}, randomSeed = {randomSeed}", 
                                    label=label_str, 
                                    ylabel="Loss",
                                    legendOn=False, # use external legend
                                    marker=""
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
