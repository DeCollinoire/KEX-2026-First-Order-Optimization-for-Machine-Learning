import numpy as np
from optimizers import sgd, nesterov, momentum, adam
from QuadraticForm import QuadraticForm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import plotPath

def testQdf():
    """
    Tests each optimizer function for finding the minimum of a quadratic form (qdf).
    """
    # Setup
    A = np.array([
         [10, 0],
         [0, 5]
         ])
    b = np.array([1, 5])
    qdf = QuadraticForm(A, b)

    expectedRoot = np.linalg.solve(qdf.A, qdf.b)
    expected_Y = qdf.evaluate_loss(expectedRoot)

    # List of optimizers
    initPos = [3, 5]
    learningRateDefault = 0.1
    sgd_optimizer = sgd.SGD(qdf, initPos, lr=learningRateDefault)
    nesterov_optimizer = nesterov.Nesterov(qdf, initPos, lr = 0.1, decayFactor = 0.9)
    momentum_optimizer = momentum.Momentum(qdf, initPos, learningRate = 0.1, decayFactor=0.3)
    adam_optimizer = adam.Adam(qdf, initPos=initPos, learningRate=1, forgettingFactorM=0.3, forgettingFactorR=0.999)
    optimizerFunctions = [sgd_optimizer, nesterov_optimizer, momentum_optimizer, adam_optimizer]

    # Testing
    for i, optimizer in enumerate(optimizerFunctions):
        # Optimize (optimizer returns position history first, then loss history)
        posHistory, lossHistory = optimizer(nr_epochs = 10)

        # Results
        print("### Testing", optimizer.__class__.__name__)
        print("Found minimum: ", optimizer.pos)
        print("Expected: ",expectedRoot)

        # Plotting
        ## Loss history
        fig = plt.figure(1)
        plt.plot(np.square(lossHistory - expected_Y), "-x", label=optimizer.__class__.__name__)
        plt.legend()

        ## Paths
        plt.figure(2)
        plt.subplot(2, 2, i+1)
        plotPath(qdf, posHistory, optimizer.__class__.__name__, center = expectedRoot, scale = 0.5)

    plt.show()

if __name__ == "__main__":
    testQdf()


