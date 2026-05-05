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

def linspace(start, end, nrPoints):
    """ Generate nrPoints evenly separated points between start and end. """
    delta = abs(end - start) / nrPoints
    return [float(f"{(start + delta * i):.3g}") for i in range(nrPoints + 1)]     # Use f-string with :.3g for 3 significant figures, then convert back to float

# Master configuration for each problem
sensitivityConfig = {
    "australian_scale": {
        "SGD": {"lr": linspace(0.001, 0.1, 50)},
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
            "fm": [0.4, 0.7, 0.9, 0.99],
            "fr": [0.4, 0.7, 0.9, 0.99]
        }
    },
    "Rosenbrock": {
        "SGD": {
            "lr": linspace(0.001, 0.002, 25),
        },
        "Nesterov": {
            "lr": [], # linspace(0.00001, 0.01, 25),
            "df": linspace(0.6, 0.95, 15)
        },
        "Adam": {
            "lr": [], #linspace(0.004, 0.02, 10),
            "fm": [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999],
            "fr": [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
        }
    }
}

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

def createVariants(lossObj, initPos, problemName="australian_scale"):
    # Extract the specific config for the current problem
    problemConfig = sensitivityConfig.get(problemName, sensitivityConfig["australian_scale"]) # default to australian_scale
    variants = {"SGD": {"lr": []},
                "Nesterov": {"lr": [], "df": []},
                "Momentum": {"lr": [], "df": []},
                "Adam": {"lr": [], "fm": [], "fr": []}
                }

    # SGD
    variants["SGD"]["lr"].append(sgd.SGD(lossObj, initPos.copy(), lr=optimizerBaseCaseConfig[problemName]["SGD"]["lr"]))
    for lr in problemConfig["SGD"]["lr"]:
        variants["SGD"]["lr"].append(sgd.SGD(lossObj, initPos.copy(), lr=lr))

    # Nesterov & Momentum (sharing Nesterov config)
    nesterovConfig = problemConfig["Nesterov"]
    variants["Nesterov"]["lr"].append(nesterov.Nesterov(lossObj, initPos.copy(), lr=optimizerBaseCaseConfig[problemName]["Nesterov"]["lr"], decayFactor=optimizerBaseCaseConfig[problemName]["Nesterov"]["decayFactor"]))
    variants["Momentum"]["lr"].append(momentum.Momentum(lossObj, initPos.copy(), learningRate=optimizerBaseCaseConfig[problemName]["Nesterov"]["lr"], decayFactor=optimizerBaseCaseConfig[problemName]["Nesterov"]["decayFactor"]))
    for lr in nesterovConfig["lr"]:
        variants["Nesterov"]["lr"].append(nesterov.Nesterov(lossObj, initPos.copy(), lr=lr, decayFactor=optimizerBaseCaseConfig[problemName]["Nesterov"]["decayFactor"]))
        variants["Momentum"]["lr"].append(momentum.Momentum(lossObj, initPos.copy(), learningRate=lr, decayFactor=optimizerBaseCaseConfig[problemName]["Nesterov"]["decayFactor"]))
    for df in nesterovConfig["df"]:
        variants["Nesterov"]["lr"].append(nesterov.Nesterov(lossObj, initPos.copy(), lr=optimizerBaseCaseConfig[problemName]["Nesterov"]["lr"], decayFactor=df))
        variants["Momentum"]["lr"].append(momentum.Momentum(lossObj, initPos.copy(), learningRate=optimizerBaseCaseConfig[problemName]["Nesterov"]["lr"], decayFactor=df))

    # Adam
    adamConfig = problemConfig["Adam"]
    variants["Adam"]["lr"].append(adam.Adam(lossObj, initPos.copy(), learningRate=optimizerBaseCaseConfig[problemName]["Adam"]["learningRate"], forgettingFactorM=optimizerBaseCaseConfig[problemName]["Adam"]["forgettingFactorM"], forgettingFactorR=optimizerBaseCaseConfig[problemName]["Adam"]["forgettingFactorR"]))
    for lr in adamConfig["lr"]: 
        variants["Adam"]["lr"].append(adam.Adam(lossObj, initPos.copy(), learningRate=lr, forgettingFactorM=optimizerBaseCaseConfig[problemName]["Adam"]["forgettingFactorM"], forgettingFactorR=optimizerBaseCaseConfig[problemName]["Adam"]["forgettingFactorR"]))
    for fm in adamConfig["fm"]:
        variants["Adam"]["fm"].append(adam.Adam(lossObj, initPos.copy(), learningRate=optimizerBaseCaseConfig[problemName]["Adam"]["learningRate"], forgettingFactorM=fm, forgettingFactorR=optimizerBaseCaseConfig[problemName]["Adam"]["forgettingFactorR"]))
    for fr in adamConfig["fr"]:
        variants["Adam"]["fr"].append(adam.Adam(lossObj, initPos.copy(), learningRate=optimizerBaseCaseConfig[problemName]["Adam"]["learningRate"], forgettingFactorM=optimizerBaseCaseConfig[problemName]["Adam"]["forgettingFactorM"], forgettingFactorR=fr))

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
    datasetFilepath = datasetMap[problemName]
    batchSize = 100000 # Use large batch size to get full batch.
    initialPosInterval = 0
    l2NormalizationOn = (problemName in ["australian", "australian_scale"])
    dim = 10 # For Rosenbrock only
    nrEpochs = 50
    printProgress = True

    # Setup
    print("Setting up...")
    np.random.seed(randomSeed)
    lossObj, initPos = setupProblem(problemName=problemName, dim=dim, datasetFilepath=datasetFilepath,
                                    initialPosInterval=initialPosInterval, randomSeed=randomSeed,
                                    batchSize=batchSize, toDense=False, l2NormalizationOn=l2NormalizationOn)  # QDF, Rosenbrock; datasetFilepath is only needed for LogReg

    # Create 'variants', which looks like this:
    # variants = { optimizerName: { hyperparam1: [values], hyperparam2: [values], ... } }
    variants = createVariants(lossObj, initPos, problemName=problemName) # Creates all variants of sgd, nesterov, momentum and adam in a dictionary
    joinedOptList = [opt for optList in variants.values() for opt in optList] # Join all optimizers into one list for training
    print(f"Setup finished!")

    # Train
    print(f"\n--- Training ---")
    train(joinedOptList, lossObj, nrEpochs=nrEpochs, printProgress=printProgress)
    print(f"Training finished.")

    # Plot
    for optClassName, hyperparamDict in variants.items():
        plt.figure(str(optClassName+"_sensitivity_test"), figsize=(10, 6))
        for hyperparamName, optimizerList in hyperparamDict.items():
            plt.subplots_adjust(left=0.1, bottom=0.05, right=0.8, top=None, wspace=None, hspace=0.9)
            lines = []
            for opt in optimizerList:
                # Pass the hyperparameter string into the label or a custom attribute
                label_str = f"{opt.__class__.__name__}: {opt.getHyperparamStr()}"
                line = plotHistoryGraph(opt.lossHistory, 
                                        title=f"{optClassName} Sensitivity, {problemName}, dim = {lossObj.xDataLength}, dataset = {datasetFilepath}, batchSize = {"fullbatch" if batchSize is None else batchSize}", 
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
