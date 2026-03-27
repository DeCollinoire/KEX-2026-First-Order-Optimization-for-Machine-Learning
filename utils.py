import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotPath(qdf, history, optimizer_name, center = [0,0], scale: float = 1):
    # Create a grid of points
    x = np.linspace(center[0] - 10 * scale, center[0] + 10 * scale, 100)
    y = np.linspace(center[1] - 10 * scale, center[1] + 10 * scale, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the loss (Z) at every point on the grid
    Z = np.array([qdf.evaluate_loss(np.array([px, py])) for px, py in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    
    # Plot the contours
    contour = plt.contour(X, Y, Z, levels=32, cmap="viridis")
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

def plotPath_3d(qdf, history, optimizer_name, center = [0,0], scale: float = 1):
    # NOTE: This will create a new figure for each call
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

def plotHistoryGraph(history, title, label, ylabel, yscale="linear"):
    plt.plot(history, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.legend()
    plt.grid()

def train(optimizerList, lossObj=None, nrEpochs=100):
    if lossObj is None:
        lossObj = optimizerList[0].lossObj
    # Start testing
    for epoch in range(1, nrEpochs + 1):
        # Shuffle batches
        lossObj.fillRandomBatchList()

        lossObj.currentBatchIndex = 0
        for batch in range(lossObj.numberOfBatches):
            for index, optimizer in enumerate(optimizerList):
                # Store history first (to include the inital position)
                optimizer.posHistory.append(optimizer.pos.copy())
                optimizer.lossHistory.append(lossObj.evaluate_loss(optimizer.pos))

                optimizer.step()

            # On to next batch for calculating gradient and so on
            lossObj.currentBatchIndex = lossObj.currentBatchIndex + 1
    return optimizerList


def setupProblem(problemName, dim = 2, datasetFilepath = "datasets/australian_scale"):
    """ Returns a loss object and an initial position
    Problems to choose from:
    - 'QDF' : Random, 2D, positive definite quadratic form
    - 'Rosenbrock' : Rosenbrock of dimension 'dim'
    - 'LogReg' : Logistic regression
    """
    # Only do imports when we get the actual problem
    if problemName == "QDF":
        from QuadraticForm import QuadraticForm
        lossObj = QuadraticForm()
        initPos = np.array([0.5, 0.5])
        return
    if problemName == "Rosenbrock":
        from Rosenbrock import Rosenbrock
        lossObj = Rosenbrock(dim)
        initPos = np.random.uniform(-10, 10, dim)
        return
    if problemName == "LogReg":
        from LogisticRegression import LogisticRegression
        from DataLoader import loadDataAsNumpyArray
        X, y = loadDataAsNumpyArray(datasetFilepath)
        lossObj = LogisticRegression(data = [X,y])
        initPos = None
    else:
        raise NotImplementedError(f"The problem '{problemName}' is not implemented.")

    return lossObj, initPos
