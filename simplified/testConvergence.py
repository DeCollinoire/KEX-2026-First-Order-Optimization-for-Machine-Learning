import numpy as np
import matplotlib.pyplot as plt

# Project files
from loss import LogisticRegression
from utils import train, plotPath, plotPath_3d, plotHistoryGraph
from DataLoader import loadDataAsNumpyArray
from optimizers import SGD, Momentum, Nesterov, Adam # type: ignore

def main():
    # Config
    randomSeed = 42
    datasetFilepath = "datasets/rcv1_train.binary" # rcv1_train.binary, australian_scale, australian - no mushrooms, stupid copilot :)
    initialPosInterval = 0.1
    nrEpochs = 15
    batchSize = 32
    toDense = False
    L2NormalizationOn = True
    printProgress = True
    divideByBatchSize = True

    # Load data
    np.random.seed(randomSeed)
    X, y = loadDataAsNumpyArray(datasetFilepath, toDense=toDense, L2NormalizationOn=L2NormalizationOn)
    lossObj = LogisticRegression((X, y))
    initPos =  np.random.uniform(-initialPosInterval, initialPosInterval, lossObj.nrFeatures)
    lossObj.divideByBatchSize = divideByBatchSize
    print("Data loaded and lossObj created.")

    # Optimizers
    optSGD = SGD(initPos, lr=100) # Samma fel för RCV1 kvarstår
    optMomentum = Momentum(initPos, lr=0.01, decayFactor=0.9)
    optNesterov = Nesterov(initPos, lr=0.01, decayFactor=0.9)
    optAdam = Adam(initPos, lr=0.01, beta1=0.9, beta2=0.999)

    # Train
    print("Starting training...")
    optimizerList = [optSGD, optMomentum, optNesterov, optAdam]
    train(optimizerList, lossObj, X, y, batchSize=batchSize, nrEpochs=nrEpochs, printProgress=printProgress)
    
    # Plot
    print("Presenting results.")
    plt.figure()
    for optimizer in optimizerList:
        plotHistoryGraph(optimizer.lossHistory, title=f"Loss History, seed = {randomSeed}, datasetFilepath = {datasetFilepath}, batchSize = {batchSize}", label=f"{optimizer.__class__.__name__}, {optimizer.getHyperparamStr()}", ylabel="Loss", yscale="linear")
    plt.show()



if __name__ == "__main__":
    main()





