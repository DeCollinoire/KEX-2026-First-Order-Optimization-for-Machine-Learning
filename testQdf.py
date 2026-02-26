import numpy as np
import matplotlib.pyplot as plt
from optimizers import sgd, nesterov, momentum, adam
from optimizers.loss.loss import QuadraticForm

def plot_path(qdf, history, optimizer_name, center = [0,0], scale: float = 1):
    # Create a grid of points
    x = np.linspace(center[0] - scale * 10, center[0] + scale * 10, 100)
    y = np.linspace(center[1] - scale * 10, center[1] + scale * 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the loss (Z) at every point on the grid
    Z = np.array([qdf.evaluate_loss(np.array([px, py])) for px, py in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    
    # Plot the contours
    contour = plt.contour(X, Y, Z, levels=20, cmap="viridis")
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot the path taken
    history = np.asarray(history)
    hx = history[:, 0]
    hy = history[:, 1]
    plt.plot(hx, hy, "r.-", label=f"{optimizer_name} Path", alpha=0.6)
    plt.plot(hx[0], hy[0], "go", label="Start")
    plt.plot(hx[-1], hy[-1], "bo", label="End")

    plt.title(f"{optimizer_name} Optimization Path on Quadratic Surface")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()
    plt.axis('square')
    plt.grid(True)

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
    initPos = [5, 5]
    learningRateDefault = 0.1
    sgd_optimizer = sgd.SGD(qdf, initPos, lr=learningRateDefault)
    nesterov_optimizer = nesterov.Nesterov(qdf, initPos, lr = 0.1, decayFactor = 0.3)
    momentum_optimizer = momentum.Momentum(qdf, initPos, learningRate = 0.1, decayFactor=0.3)
    adam_optimizer = adam.Adam(qdf, initPos=initPos, learningRate=1, forgettingFactorM=0.3, forgettingFactorR=0.999)
    optimizerFunctions = [sgd_optimizer, nesterov_optimizer, momentum_optimizer, adam_optimizer]

    # Testing
    for i, optimizer in enumerate(optimizerFunctions):
        # Optimize (optimizer returns position history first, then loss history)
        posHistory, lossHistory = optimizer(nr_epochs = 25)

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
        plot_path(qdf, posHistory, optimizer.__class__.__name__, center = expectedRoot, scale = 0.5)

    plt.show()


if __name__ == "__main__":
    testQdf()


