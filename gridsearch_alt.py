import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Project files
from QuadraticForm import QuadraticForm
from optimizers.momentum import Momentum
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam
from utils import plotHistoryGraph, train
from DataLoader import loadDataAsNumpyArray
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock
from utils import setupProblem
import itertools

import mplcursors




def gridsearchAlt(lossObj, initPos, optClass, groupOfHyperParameterlists, nrEpochs=50, datasetFilepath=""):
    """
    groupOfHyperParameterlists is structured as follows

    [
        lr variants = []
        ...
    ]

    :param optClass:
    :param groupOfHyperParameterlists:
    :return:
    """

    plt.figure(figsize=(10, 8))
    #plt.figure(figsize=(12, 8), layout='constrained')

    for combination in itertools.product(*groupOfHyperParameterlists):
        opt = optClass(lossObj, initPos, *combination)
        train([opt], lossObj=lossObj, nrEpochs=nrEpochs)

        plotHistoryGraph(
            opt.lossHistory,
            title=f"Loss: {opt.lossObj.__class__.__name__}, Dataset: {datasetFilepath}",
            label=f"{opt.__class__.__name__}, {opt.getHyperparamStr()}",
            ylabel="Loss"
        )

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # plt.legend()
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.9)


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

    return

def main():
    # Setup problem
    datasetFilepath = "datasets/australian_scale"  # rcv1_train.binary, Rosenbrock or australian_scale
    print("Set up problem: Begun")
    problemName = "LogReg"
    lossObj, initPos = setupProblem(problemName=problemName, dim=10, datasetFilepath=datasetFilepath,
                                    initialPosInterval=0, randomSeed=10,
                                    batchSize=1000000, toDense=False, l2NormalizationOn=True)  # QDF, Rosenbrock; datasetFilepath is only needed for LogReg

    #Rosenbrock
    # randomSeed=10
    # datasetFilepath = "Rosenbrock" # This is also used to show the name in the plot title
    # lossObj, initPos = setupProblem("Rosenbrock", dim=10, randomSeed = randomSeed)
    #
    # lossObj, initPos = setupProblem("QDF", randomSeed = randomSeed)
    #
    # print("Set up problem: Finished")
    # print("initpos" + str(initPos))



    gridNrEpochs=50

    # Momentum setup
    momentumLrList = [0.002, 0.035, 0.02,0.06,0.07, 0.08, 0.1]
    momentumDfList = [0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    # gridsearchAlt(lossObj=lossObj, initPos=initPos, optClass=momentum.Momentum, groupOfHyperParameterlists=[momentumDfList, momentumLrList], nrEpochs=gridNrEpochs)


    # Nesterov setup
    nesterovLrList = [0.01, 0.02, 0.7, 0.8, 0.85, 0.9]
    nesterovDfList = [0.1, 0.2, 0.5, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    # gridsearchAlt(lossObj=lossObj, initPos=initPos,optClass=nesterov.Nesterov, groupOfHyperParameterlists=[nesterovLrList, nesterovDfList], nrEpochs=gridNrEpochs)


    # Adam setup
    adamLrList = [0.002, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1]
    adamFfMList = [0.2, 0.8, 0.9, 0.91, 0.95]
    adamFfRList = [0.7, 0.8, 0.99, 0.999]

    gridsearchAlt(lossObj=lossObj, initPos=initPos, optClass=adam.Adam, groupOfHyperParameterlists=[adamLrList, adamFfMList, adamFfRList], nrEpochs=gridNrEpochs)



if __name__ == "__main__":
    main()

