import numpy as np
import matplotlib.pyplot as plt
import copy
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam
from QuadraticForm import QuadraticForm
from utils import plotHistoryGraph, train

#Incomplete

class OptimizerGroup:

    def __init__(self, optimizerBaseCase: Optimizer):
        self.optimizerBaseCase = optimizerBaseCase

        #To get the right amount of variant lists. Floats will be replaced with lists
        # [
        #     "hyperparemetername": [opt1, opt2...] #With that specific hyperparameter altered
        #
        # ]
        self.optVariantsDictList = copy.deepcopy(optimizerBaseCase.getHyperparamDict())


    def getAllOptimizers(self):
        """
        Returns a list with all optimizers contained within this OptimizerGroup
        :return:
        """
        allOpt = [self.optimizerBaseCase]
        for optimizerlist in self.optVariantsDictList:
            allOpt.extend(optimizerlist.values())
        return allOpt



def createVariants(lossObj, initPos):

    #Incomplete
    lrAdam = 0.5
    forgettingFactorMAdam = 0.9
    forgettingFactorRAdam = 0.999
    baseCaseAdam = adam.Adam(lossObj, initPos, learningRate=lrAdam, forgettingFactorM=forgettingFactorMAdam, forgettingFactorR=forgettingFactorRAdam)

    optimizerGroupAdam = OptimizerGroup(optimizerBaseCase=baseCaseAdam)

    #Add variants to optimizer group object
    #***


    optSGD = sgd.SGD(lossObj, initPos, lr=0.5)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.5, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.5, decayFactor=0.9)

    return [optimizerGroupAdam]



def main():
    #Incomplete
    lossObj = QuadraticForm()
    initPos = False  #Incomplete

    #Create a list of Optimizer group objects. Each of which contains the base case and variants.
    optimizerGroupList = createVariants(lossObj=lossObj, initPos=initPos)

    allOptimizers = []
    for optGroup in optimizerGroupList:
        allOptimizers.append(optGroup.getAllOptimizers())

    train(allOptimizers, nrEpochs=100)

    #





if __name__ == "__main__":
    main()