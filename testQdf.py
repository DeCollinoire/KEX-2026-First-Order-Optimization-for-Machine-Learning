import numpy as np
import matplotlib.pyplot as plt
from optimizers.optimizer import SGD, Nesterov
from optimizers.loss import QuadraticForm

def plot_optimization(qdf, history, optimizer_name):
    # Create a grid of points
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the loss (Z) at every point on the grid
    Z = np.array([qdf.loss(np.array([px, py])) for px, py in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    
    # Plot the contours
    contour = plt.contour(X, Y, Z, levels=20, cmap="viridis")
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot the path taken
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
    tol = 0.01      # Tolerance for when loss is "small enough"
    A = np.array([
        [19, 0],
        [0, 5]
        ])
    b = np.array([1, 5])
    qdf = QuadraticForm(A, b)

    expectedRoot = np.linalg.solve(qdf.A, qdf.b)
    expected_Y = qdf.loss(expectedRoot)
    

    # List of optimizers
    initParams = [2, 5]
    sgd_optimizer = SGD(initParams, lr=0.1)
    nesterov_optimizer = Nesterov(initParams, lr = 0.1, gamma = 0.1)
    optimizerFunctions = [sgd_optimizer, nesterov_optimizer]

    # Testing
    for i, optimizer in enumerate(optimizerFunctions):
        lossHistory, paramsHistory = optimizer(qdf)    # data not needed for quadratic forms

        # Results
        print("### Testing", optimizer.__class__.__name__)
        print("Found minimum: ", optimizer.params)
        print("Expected: ",expectedRoot)

        # Plotting
        ## Loss history
        fig = plt.figure(1)
        plt.plot(np.square(lossHistory - expected_Y), "-x", label=optimizer.__class__.__name__)
        plt.legend()

        ## Paths
        plt.figure(2)
        plt.subplot(2, 2, i%2+1)
        plot_optimization(qdf, paramsHistory, optimizer.__class__.__name__)

    plt.show()


if __name__ == "__main__":
    testQdf()


