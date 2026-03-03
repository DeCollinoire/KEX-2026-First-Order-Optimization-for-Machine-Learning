import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from optimizers import sgd, nesterov, momentum, adam
from QuadraticForm import QuadraticForm



def plot_path(qdf, history, optimizer_name, center = [0,0], scale: float = 1):
    # Create a grid of points
    x = np.linspace(-10 * scale, 10 * scale, 100)
    y = np.linspace(-10 * scale, 10 * scale, 100)
    w = np.linspace(-10 * scale, 10 * scale, 100)
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

def plot_path_3d(qdf, history, optimizer_name, center = [0,0], scale: float = 1):
    # NOTE: This will create a new figure for each call, so one needs to modify the test loop
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid of points
    x = np.linspace(-10 * scale, 10 * scale, 100)
    y = np.linspace(-10 * scale, 10 * scale, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z (loss) for the surface
    Z = np.array([qdf.evaluate_loss(np.array([px, py])) for px, py in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    
    # Plot the 3D surface
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, linewidth=0, antialiased=True)
    
    # Process history for the path
    history = np.asarray(history)
    hx = history[:, 0]
    hy = history[:, 1]
    # Calculate Z values for the path so it sits on the surface
    hz = np.array([qdf.evaluate_loss(p) for p in history])
    
    # Plot the path taken (slightly offset Z to prevent clipping into the surface)
    ax.plot(hx, hy, hz + 0.1, "r.-", label=f"{optimizer_name} Path", markersize=5, zorder=10)
    ax.scatter(hx[0], hy[0], hz[0] + 0.2, color="green", s=50, label="Start", zorder=11)
    ax.scatter(hx[-1], hy[-1], hz[-1] + 0.2, color="blue", s=50, label="End", zorder=11)

    ax.set_title(f"{optimizer_name} Optimization Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Loss")
    
    plt.legend()

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
        plot_path_3d(qdf, posHistory, optimizer.__class__.__name__, center = expectedRoot, scale = 0.5)

    plt.show()


if __name__ == "__main__":
    testQdf()


