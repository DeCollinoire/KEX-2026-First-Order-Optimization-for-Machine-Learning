import numpy as np

# Project files
from optimizers import sgd, momentum, nesterov, adam
from QuadraticForm import QuadraticForm
from Rosenbrock import Rosenbrock
from LogisticRegression import LogisticRegression
from DataLoader import loadDataAsNumpyArray

# Plotting
from utils import plotPath, plotHistoryGraph, plotPath_3d, train
import matplotlib.pyplot as plt


def testConvergenceSingleOptimizer(optimizer, tol, nr_epochs):
    posHistory, lossHistory = optimizer(nr_epochs)

    # Estimate errors
    minima = optimizer.lossObj.minima()
    if minima is not None:
        errors = np.linalg.norm(posHistory - minima, axis=1) # Estimate error as distance to a known minima
    else:
        errors = np.linalg.norm(np.diff(posHistory, axis=0), axis=1)  # Estimate error as change in position

    # Convergence ratios
    conv_ratios = errors[1:] / (errors[:-1] + 1e-15)

    # Find stopping point based on position change
    under_tol = np.where(errors < tol)[0]
    n_steps = under_tol[0] + 1 if under_tol.size > 0 else nr_epochs

    # Estimate convergence order q
    q = estimateOrder(errors[:n_steps]) if n_steps > 3 else None

    return conv_ratios, n_steps, q

def estimateOrder(errors):
    # Filter out zeros or negative values to avoid NaN in log
    errors = errors[errors > 1e-15]
    if len(errors) < 3: return None

    # Calculate log of successive error ratios
    # log(e_{k+1}/e_k)
    log_ratios = np.log(errors[1:] / errors[:-1])

    # q = log(e_{k+1}/e_k) / log(e_k/e_{k-1})
    numerator = log_ratios[1:]
    denominator = log_ratios[:-1]

    # Avoid division by zero (happens if convergence is perfectly linear)
    q_estimates = numerator / (denominator + 1e-15)

    # Return average of recent stable estimates, excluding Infs/NaNs
    valid_q = q_estimates[np.isfinite(q_estimates)]
    return np.mean(valid_q[-10:]) if valid_q.size > 0 else None

def testConvergenceBatched(optimizerList, lossObj, nr_epochs = 100):
    # Setup
    analyze = False
    minima = lossObj.minima()
    errors = [[] for _ in range(len(optimizerList))]
    hasConvergedCheckList = [False for _ in range(len(optimizerList))]

    # Start testing
    for epoch in range(1, nr_epochs + 1):
        # Save the history per epoch
        for optimizer in optimizerList:
            optimizer.savePosition()

        # Shuffle batches
        lossObj.fillRandomBatchList()
        
        # Step each optimizer in parallel (using the same batch).
        for batch in range(lossObj.numberOfBatches):
            for index, optimizer in enumerate(optimizerList):
                if hasConvergedCheckList[index] == False:
                    optimizer.step()

                    # NOTE: Skip calculations due to issues with minima. Potentially change to check convergence in loss values instead of absolute error from minima.                
                    if analyze:
                        # Calculate error and save it
                        error = np.linalg.norm(optimizer.pos - minima)
                        errors[index].append(error)

                        # Check convergence
                        if error < 1e-4:
                            hasConvergedCheckList[index] = True
                            N_steps = epoch * (lossObj.numberOfBatches - 1) + batch

                            # Present
                            print(f"{optimizer.__class__.__name__} converged in {N_steps} steps and {epoch} epochs.")
                            print(f"Optimizer: {optimizer.__class__.__name__}")
                            print(f"Final position: {optimizer.pos}, Minima: {minima}")
                            print(f"Number of steps to reach tolerance: {N_steps}")
                            # print(f"Convergence ratios: {conv_ratios}")
                            # print(f"Estimated convergence order: {q}")
                    
            # Step to the next batch
            lossObj.currentBatchIndex = lossObj.currentBatchIndex + 1

def main():
    # Setup problem
    X, y = loadDataAsNumpyArray("datasets/australian_scaled")
    lossObj = LogisticRegression(data=[X, y])

    # Setup optimizers
    nrFeatures = lossObj.xDataLength
    initPos = np.random.uniform(-10, 10, nrFeatures) # if the same position is wanted, set the seed using: np.random.seed(...)

    optSGD = sgd.SGD(lossObj, initPos, lr=0.1)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)
    
    # Run the test
    optimizerList=[optSGD, optNesterov, optMomentum, optAdam]
    testConvergenceBatched(optimizerList, lossObj, nr_epochs=25)

    # Plotting & Presenting
    plt.figure()
    for optimizer in optimizerList:
        plotHistoryGraph(optimizer.lossHistory, f"Loss history for {optimizer.__class__.__name__}", f"{optimizer.__class__.__name__}, {optimizer.getHyperparamStr()}", "Loss")
    plt.show()

if __name__ == "__main__":
    main()