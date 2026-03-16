import numpy as np
import matplotlib.pyplot as plt
from optimizers.optimizer import Optimizer
from optimizers import sgd, nesterov, momentum, adam
from QuadraticForm import QuadraticForm
from utils import plotHistoryGraph



class optimizerGroup:
    def __init___(self, optimizer: Optimizer):
        self.optimizerBaseCase = optimizer
        self.optVariants = self.createVariants(optimizer, [0.1, 0.2])

    def createVariants(self, optimizerBaseCase, hyperparamName hyperparamValues):
        """
        "lr": [opt_h1, opt_h2, opt_h3] # hyperparam 1
        """
        hyperparamValues = self.optimizerBaseCase.getHyperparamDict()
        optimizerList = []
        for value in hyperparamValues:
            opt = optimizerBaseCase.copy()
            setattr(opt, hyperparamName, value) ## NOTE: hyperparamName is a parameter for this function, needs to be changed
            optimizerList.append(opt)
        return optimizerList


def train(optimizerList, nrEpochs=100):
    lossObj = optimizerList[0].lossObj

    # Start testing
    for epoch in range(1, nrEpochs + 1):
        # Shuffle batches
        lossObj.fillRandomBatchList()

        for batch in range(lossObj.numberOfBatches): # len(lossObj.randomBatchList) = lossObj.numberOfBatches
            for index, optimizer in enumerate(optimizerList):
                optimizer.step()
                optimizer.posHistory.append(optimizer.pos.copy())
                optimizer.lossHistory.append(lossObj.evaluate_loss(optimizer.pos))

            # On to next batch for calculating gradient and so on
            lossObj.currentBatchIndex = lossObj.currentBatchIndex + 1
    return

def calculateMeanDifference(base, lossHistories):
    """
    Gets the "final" loss history value for each loss history and calculates the mean between them and the base value.
    """
    pass

def test_hyperparameter_sensitivity(optimizer: Optimizer, nrEpochs=100):
    """"
    Results will be a dictionary on the following format:
    results = {
        "h1": [loss1, loss2] # hyperparam1
        "h1": [loss1, loss2] # hyperparam2
    }
    """

    hyperparamsDict = optimizer.getHyperparamDict()
    results = hyperparamsDict.copy()    # Values will be overwritten in the loop below

    # Test each hyperparameter
    adjustmentFactor = [0.001, 0.01, 0.1, 1]
    for hyperparamName, baseval in hyperparamsDict.items():
        # Test different values of the hyperparameter
        results[hyperparamName] = []
        for factor in adjustmentFactor:
            # Adjust the basevalue in the optimizer
            optTemp = optimizer
            setattr(optimizer, hyperparamName, baseval * factor)
            
            # Run the optimization and get the results
            posHistory, lossHistory = optimizer(nr_epochs=nrEpochs)
            results[hyperparamName].append(lossHistory)

        # Reset hyperparameter to base value
        setattr(optimizer, hyperparamName, baseval)

    return results, adjustmentFactor

def main():
    A = np.array([
        [19, 0],
        [0, 5]
        ])
    b = np.array([1, 5])
    lossObj = QuadraticForm(A, b)
    initPos = np.array([1.0, 1.0])

    # Setup of optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr=0.5)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.5, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.5, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.5, forgettingFactorM=0.9, forgettingFactorR=0.999)

    optimizerList = [optSGD, optNesterov, optMomentum, optAdam]
    for opt in optimizerList:
        results, adjustmentFactors = test_hyperparameter_sensitivity(opt, nrEpochs=5) # Results is a dictionary that maps each hyperparamater name to a list of loss histories, one for each hyperparameter value.

        print(f" \n Optimizer: {opt.__class__.__name__}")
        for hyperparamName, lossHistories in results.items(): 
            print(f"\t {hyperparamName}: {lossHistories}")
            plt.figure()
            for i in range(len(lossHistories)):
                #plt.subplot(2,2, i+1)
                plotHistoryGraph(lossHistories[i], f"{opt.__class__.__name__} Loss History \n {hyperparamName} = {adjustmentFactors[i]}", "Loss")
            plt.show()

def mainAlt():
    A = np.array([
        [19, 0],
        [0, 5]
        ])
    b = np.array([1, 5])
    lossObj = QuadraticForm(A, b)
    initPos = np.array([1.0, 1.0])
    
    # Test learning rate
    optSGD = sgd.SGD(lossObj, initPos, lr=0.5)
    optimizerListSGD = createVariants(optSGD, "lr", [0.1, 0.2, 0.5])

    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.5, decayFactor=0.9)
    """

    """
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.5, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.5, forgettingFactorM=0.9, forgettingFactorR=0.999)

    optimizerList = [optSGD, optNesterov, optMomentum, optAdam]
    for opt in optimizerList:
        results = test_hyperparameter_sensitivity(opt, nrEpochs=5) # Results is a dictionary that maps each hyperparamater name to a list of loss histories, one for each hyperparameter value.

        print(f" \n Optimizer: {opt.__class__.__name__}")
        for name, lossHistory in results.items():
            print(f"\t {name}: {lossHistory}")
            plt.figure()
            #plt.subplot(2,2, i+1)
            plotHistoryGraph(lossHistory, f"{opt.__class__.__name__} Loss History \n {opt.getHyperparamStr()}", "LOss")                
            plt.show()




if __name__ == "__main__": 
    main()
