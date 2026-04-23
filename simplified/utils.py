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
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.grid(True)
    plt.legend()
    plt.grid()

def _iter_batches(X, y, batch_size):
    """ Generator that yields batches from X and y """
    for start in range(0, len(y), batch_size):
        yield X[start:start + batch_size], y[start:start + batch_size]

def train(optimizerList, lossObj, X, y, batchSize=None, nrEpochs=100, printProgress=False):
    nrSamples = lossObj.nrSamples

    if batchSize is None:
        batchSize = nrSamples
        print(f"Full batch activated: batchSize = {nrSamples}")

    # Save initial position
    for optimizer in optimizerList:
        optimizer.savePosition(lossObj)

    for epoch in range(nrEpochs):
        # Shuffle data. Potentially use lossObj.shuffleData.
        indices = np.random.permutation(nrSamples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Step optimizers for each batch
        for X_batch, y_batch in _iter_batches(X_shuffled, y_shuffled, batchSize):
            lossObj.setCurrentBatch(X_batch, y_batch)
            for optimizer in optimizerList:
                optimizer.step(lossObj) # step internally calls eval_gradient, and lossObj then internally gets the current batch

        # Save positions
        for optimizer in optimizerList:
            optimizer.savePosition(lossObj)

        if printProgress:
            print(f"Epoch {epoch+1}/{nrEpochs} completed.")
