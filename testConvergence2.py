import numpy as np

# Project files
from optimizers import sgd, momentum, nesterov, adam
from QuadraticForm import QuadraticForm
from Rosenbrock import Rosenbrock
from LogisticRegression import LogisticRegression

# Plotting
from utils import plotPath, plotHistoryGraph, plotPath_3d
import matplotlib.pyplot as plt


def testConvergence(optimizer, tol, nr_epochs):
    posHistory, lossHistory = optimizer(nr_epochs)

    # Estimate errors as distance between two successive positions or from the minima (extrema) if known
    minima = optimizer.lossObj.minima()
    if minima is not None:
        errors = np.linalg.norm(posHistory - minima, axis=1)
    else:
        errors = np.linalg.norm(np.diff(posHistory, axis=0),
                                axis=1)  # Estimate error as change in position (not ideal, but can indicate convergence behavior)

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


def main():
    # Setup
    dim = 5
    lossObj = Rosenbrock(dim)
    minima = lossObj.minima()
    Plot3d = False
    nrEpochs = 1000

    np.random.seed(42)  # Remove to get different random initial positions each run
    initPos = np.random.randint(-10, 10, size=dim)
    if initPos[0] == 1 and initPos[1] == 1:
        initPos[0] += 1  # Avoid starting at the minima

    # Setup of optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr=0.005)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.001, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.001, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.9)

    # Test each optimizer and present
    for index, optimizer in enumerate([optSGD, optNesterov, optMomentum, optAdam]):
        conv_ratios, N_steps, q = testConvergence(optimizer, tol=1e-4, nr_epochs=nrEpochs)

        # Present
        print(f"\n---- Optimizer: {optimizer.__class__.__name__} ----")
        print(
            f"Final position: {optimizer.pos}, True Minima: {minima}, Error: {np.linalg.norm(optimizer.pos - minima)}")
        print(f"Number of steps to reach tolerance: {N_steps}")
        print(f"Estimated convergence order: {q}")
        print(f"Convergence ratios of last 10 iterations: {conv_ratios[-10:]}")

        # Plotting
        plt.figure(1)
        plt.subplot(2, 2, index + 1)
        plotPath(optimizer.lossObj, optimizer.posHistory, optimizer.__class__.__name__, scale=3)
        plt.plot(minima[0], minima[1], "r*", markersize=10, label="True Minima") if minima is not None else None
        plt.legend()

        plt.figure(2)
        plt.subplot(2, 2, index + 1)
        plotHistoryGraph(conv_ratios, f"Convergence Ratios for {optimizer.__class__.__name__}", "Convergence Ratio",
                         yscale="linear")

        plt.figure(3)
        plt.subplot(2, 2, index + 1)
        plotHistoryGraph(optimizer.lossHistory, f"Loss History for {optimizer.__class__.__name__}", "Loss",
                         yscale="linear")

        # 3D plot
        if Plot3d:
            plotPath_3d(lossObj, optimizer.posHistory, f'Surface for {optimizer.__class__.__name__}', center=minima,
                        scale=3)
            plt.plot(minima[0], minima[1], 0, "r*", markersize=10, label="True Minima") if minima is not None else None
            plt.legend()

    plt.show()


def convergenceHandlerBatchVersion(initPos, optimizerList = [], lossObj = None):
    # NOTE: DOES NOT WORK YET. Maybe works now?
    # Config
    nr_epochs = 100
    #lossObj = QuadraticForm()
    minima = lossObj.minima()
    #initPos = [2, 5]

    # Setup of optimizers
    # optSGD = sgd.SGD(lossObj, initPos, lr=0.01)
    # optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    # optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
    # optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)

    # Start testing
    #errors = [[], [], [], []]
    errors = [[] for _ in range(len(optimizerList))]
    convergeList = [False for _ in range(len(optimizerList))]

    for epoch in range(1, nr_epochs + 1):
        # NOTE: Shuffle batches here
        lossObj.fillRandomBatchList()


        for batch in range(lossObj.numberOfBatches): #len(lossObj.randomBatchList) = lossObj.numberOfBatches
            for index, optimizer in enumerate(optimizerList):

                if convergeList[index] == False:

                    optimizer.step()

                    # Check convergence
                    error = np.linalg.norm(optimizer.pos - minima)  # check if minima exist first
                    errors[index].append(error)

                    if error < 1e-4:
                        #print(f"{optimizer.__class__.__name__} converged in {epoch} steps.") #Batches instead of epoch? One batch is used for each step

                        N_steps = epoch * (lossObj.numberOfBatches - 1) + batch
                        print(f"{optimizer.__class__.__name__} converged in {N_steps} steps and {epoch} epochs.")
                        #N_steps = epoch

                        # Present
                        print(f"Optimizer: {optimizer.__class__.__name__}")
                        print(f"Final position: {optimizer.pos}, Minima: {minima}")
                        # print(f"Convergence ratios: {conv_ratios}")
                        print(f"Number of steps to reach tolerance: {N_steps}")
                        # print(f"Estimated convergence order: {q}")

                        convergeList[index] = True


            #On to next batch for calculating gradient and so on
            lossObj.currentBatch = lossObj.currentBatch + 1

        # for index, optimizer in enumerate([optSGD, optNesterov, optMomentum, optAdam]):
        #     optimizer.step()
        #
        #     # Check convergence
        #     error = np.linalg.norm(optimizer.pos - minima)  # check if minima exist first
        #     errors[index].append(error)
        #
        #     if error < 1e-4:
        #         print(f"{optimizer.__class__.__name__} converged in {epoch} steps.")
        #         N_steps = epoch
        #
        #         # Present
        #         print(f"Optimizer: {optimizer.__class__.__name__}")
        #         print(f"Final position: {optimizer.pos}, Minima: {minima}")
        #         # print(f"Convergence ratios: {conv_ratios}")
        #         print(f"Number of steps to reach tolerance: {N_steps}")
        #         # print(f"Estimated convergence order: {q}")


def mainAlt():

    #Need to correct details such as initPos, data and lossObj LogisticRegression complaint even though one is a version of the other
    lossObj = LogisticRegression
    initPos = None

    optSGD = sgd.SGD(lossObj, initPos, lr=0.01)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr=0.1, decayFactor=0.9)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate=0.1, decayFactor=0.9)
    optAdam = adam.Adam(lossObj, initPos, learningRate=0.1, forgettingFactorM=0.9, forgettingFactorR=0.999)

    convergenceHandlerBatchVersion(initPos, optimizerList=[optSGD, optNesterov, optMomentum, optAdam])

if __name__ == "__main__":
    main()