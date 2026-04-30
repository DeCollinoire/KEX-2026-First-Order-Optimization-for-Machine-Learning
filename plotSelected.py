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


# Rosenbrock optimizer setups
def setupSGDRosenbrock(lossObj, initPos):
    sgdList = []
    sgdList.append(sgd.SGD(lossObj, initPos, lr=0.002))
    return sgdList

def setupMomentumRosenbrock(lossObj, initPos):
    momentumList =[]
    momentumList.append(momentum.Momentum(lossObj, initPos, learningRate=0.002, decayFactor=0.3))
    momentumList.append(momentum.Momentum(lossObj, initPos, learningRate=0.002, decayFactor=0.6))
    return momentumList

def setupNesterovRosenbrock(lossObj, initPos):
    nesterovList = []
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.002, decayFactor=0.9))
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.002, decayFactor=0.6))
    return nesterovList

def setupAdamRosenbrock(lossObj, initPos):
    adamList = []
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.06,forgettingFactorR=0.999))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.0108, forgettingFactorM=0.007, forgettingFactorR=0.999))
    return adamList


# Rcv1 optimizer setups
def setupSGDRcv1(lossObj, initPos):
    sgdList = []
    sgdList.append(sgd.SGD(lossObj, initPos, lr=0.035))
    return sgdList

def setupMomentumRcv1(lossObj, initPos):
    momentumList =[]
    momentumList.append(momentum.Momentum(lossObj, initPos, learningRate=0.07, decayFactor=0.9))
    return momentumList

def setupNesterovRcv1(lossObj, initPos):
    nesterovList = []
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.07, decayFactor=0.7))
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.07, decayFactor=0.8))
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.07, decayFactor=0.85))
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.07, decayFactor=0.9))
    return nesterovList

def setupAdamRcv1(lossObj, initPos):
    adamList = []
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.2, forgettingFactorM=0.91, forgettingFactorR=0.999))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.5, forgettingFactorM=0.5, forgettingFactorR=0.999))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.5, forgettingFactorM=0.2, forgettingFactorR=0.999))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.15, forgettingFactorM=0.8, forgettingFactorR=0.8))
    return adamList


# Australian optimizer setups
def setupSGDAustralian_scale(lossObj, initPos):
    sgdList = []
    sgdList.append(sgd.SGD(lossObj, initPos, lr=0.035))
    return sgdList

def setupMomentumAustralian_scale(lossObj, initPos):
    momentumList =[]
    momentumList.append(momentum.Momentum(lossObj, initPos, learningRate=0.035, decayFactor=0.5))
    momentumList.append(momentum.Momentum(lossObj, initPos, learningRate=0.02, decayFactor=0.5))
    momentumList.append(momentum.Momentum(lossObj, initPos, learningRate=0.02, decayFactor=0.6))
    momentumList.append(momentum.Momentum(lossObj, initPos, learningRate=0.025, decayFactor=0.5))
    return momentumList

def setupNesterovAustralian_scale(lossObj, initPos):
    nesterovList = []
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.035, decayFactor=0.8))
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.035, decayFactor=0.4))
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.035, decayFactor=0.5))
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.02, decayFactor=0.7))
    nesterovList.append(nesterov.Nesterov(lossObj, initPos, lr=0.025, decayFactor=0.7))
    return nesterovList

def setupAdamAustralian_scale(lossObj, initPos):
    adamList = []
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.2, forgettingFactorM=0.8, forgettingFactorR=0.8))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.2, forgettingFactorM=0.9, forgettingFactorR=0.999))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.15, forgettingFactorM=0.8, forgettingFactorR=0.7))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.15, forgettingFactorM=0.8, forgettingFactorR=0.8))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.3, forgettingFactorM=0.8, forgettingFactorR=0.8))
    adamList.append(adam.Adam(lossObj, initPos, learningRate=0.4, forgettingFactorM=0.6, forgettingFactorR=0.4))
    return adamList

def main():
    # Setup problem
    datasetFilepath = "datasets/australian_scale"  # rcv1_train.binary or australian_scale
    print("Set up problem: Begun")
    problemName = "LogReg"
    lossObj, initPos = setupProblem(problemName=problemName, dim=10, datasetFilepath=datasetFilepath,
                                    initialPosInterval=0, randomSeed=10,
                                    batchSize=1000000, toDense=False,
                                    l2NormalizationOn=True)  # QDF, Rosenbrock; datasetFilepath is only needed for LogReg

    # Rosenbrock
    # datasetFilepath = "Rosenbrock" # This is also used to show the name in the plot title
    # lossObj, initPos = setupProblem("Rosenbrock", dim=10, randomSeed = randomSeed)

    # lossObj, initPos = setupProblem("QDF", randomSeed = randomSeed)



    print("Set up problem: Finished")
    print("initpos" + str(initPos))

    optimizerList = []
    optimizerListAustralian_scale = []
    optimizerListRcv1 = []
    optimizerListRosenbrock = []

    # Choose optimizer and dataset setup
    # Optimizer choice
    # Rosenbrock
    optimizerListRosenbrock.extend(setupSGDRosenbrock(lossObj=lossObj, initPos=initPos))
    optimizerListRosenbrock.extend(setupMomentumRosenbrock(lossObj=lossObj, initPos=initPos))
    optimizerListRosenbrock.extend(setupNesterovRosenbrock(lossObj=lossObj, initPos=initPos))
    optimizerListRosenbrock.extend(setupAdamRosenbrock(lossObj=lossObj, initPos=initPos))

    # australian_scale
    optimizerListAustralian_scale.extend(setupSGDAustralian_scale(lossObj=lossObj, initPos=initPos))
    optimizerListAustralian_scale.extend(setupMomentumAustralian_scale(lossObj=lossObj, initPos=initPos))
    optimizerListAustralian_scale.extend(setupNesterovAustralian_scale(lossObj=lossObj, initPos=initPos))
    optimizerListAustralian_scale.extend(setupAdamAustralian_scale(lossObj=lossObj, initPos=initPos))

    # rcv1
    optimizerListRcv1.extend(setupSGDRcv1(lossObj=lossObj, initPos=initPos))
    optimizerListRcv1.extend(setupMomentumRcv1(lossObj=lossObj, initPos=initPos))
    optimizerListRcv1.extend(setupNesterovRcv1(lossObj=lossObj, initPos=initPos))
    optimizerListRcv1.extend(setupAdamRcv1(lossObj=lossObj, initPos=initPos))


    # dataset setup choice (choose one). Remember to change dataset above.
    optimizerList.extend(optimizerListAustralian_scale)
    # optimizerList.extend(optimizerListRcv1)
    # optimizerList.extend(optimizerListRosenbrock)

    train(optimizerList=optimizerList, lossObj=lossObj, nrEpochs=100)

    for opt in optimizerList:
        plotHistoryGraph(
            opt.lossHistory,
            title=f"Loss: {opt.lossObj.__class__.__name__}, Dataset: {datasetFilepath}",
            label=f"{opt.__class__.__name__}, {opt.getHyperparamStr()}",
            ylabel="Loss"
        )


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

    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

if __name__ == "__main__":
    main()