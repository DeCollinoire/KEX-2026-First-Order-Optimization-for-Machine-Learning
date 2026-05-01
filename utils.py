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

def plotHistoryGraph(history, title, label, ylabel, yscale="linear", legendOn=True, marker="o"):
    # Ensure you use the comma (line,) to unpack the list returned by plot
    line, = plt.plot(history, marker=marker, label=label)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.grid(True)
    if legendOn:
        plt.legend()
    return line

def train(optimizerList, lossObj=None, nrEpochs=50, printProgress=False):
    if lossObj is None:
        lossObj = optimizerList[0].lossObj # Assume all optimizers have the same loss object, so we can take it from the first one

    # Start testing
    for epoch in range(0, nrEpochs):
        # Save the history per epoch
        for optimizer in optimizerList:
            optimizer.savePosition()
        print(f"Epoch {epoch+1}/{nrEpochs} completed. \r") if printProgress else None


        # Shuffle batches
        lossObj.fillRandomBatchList()
        
        # Step each optimizer in parallel (using the same batch).
        for batch in range(lossObj.numberOfBatches):
            for index, optimizer in enumerate(optimizerList):
                optimizer.step()

            # Step to the next batch
            lossObj.currentBatchIndex = lossObj.currentBatchIndex + 1
    return optimizerList

def train_external_batching(optimizerList, lossObj, X, y, batchSize=None, nrEpochs=100, printProgress=False):
    """ 
    This version externalizes batching from the lossObj.
    """
    nrSamples = X.shape[0]
    nrFeatures = X.shape[1]
    if batchSize is None:
        batchSize = nrSamples # If no batch size is provided, use the whole dataset as one batch (i.e., full-batch gradient descent)
    numberOfBatches = int(np.ceil(nrSamples / batchSize))

    # Start testing
    for epoch in range(0, nrEpochs):
        # Save position for each epoch
        for optimizer in optimizerList:
            optimizer.savePosition()
        print(f"Epoch {epoch+1}/{nrEpochs} completed. \r") if printProgress else None

        # Shuffle data
        indices = np.random.permutation(nrSamples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for batch in range(numberOfBatches):
            # Create batch
            start_idx = batch * batchSize
            end_idx = min((batch + 1) * batchSize, nrSamples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Set current batch in the loss object (to avoid passing the batch to the optimizers). This is necessary since the optimizers call lossObj.evaluate() without passing the batch, so the lossObj needs to know which batch to use internally. An alternative is to pass the lossObj and the batch to the optimizers directly
            lossObj.setCurrentBatch(X_batch, y_batch)

            # Step the optimizers
            for optimizer in optimizerList:
                optimizer.step()



def setupProblem(problemName, dim=10, datasetFilepath="datasets/australian_scaled", randomSeed=0, initialPosInterval=0.1, batchSize = 64, toDense = False, l2NormalizationOn=False):
    """ 
    Returns a loss object and an initial position
    Problems to choose from:
    - 'QDF' : Random, 2D, positive definite quadratic form
    - 'Rosenbrock' : Rosenbrock of dimension 'dim', by default dim=10. 'dim' is only used for the analytical minimum; (generalized) RB can use input of any length.
    - 'LogReg' : Logistic regression on the chosen dataset at 'datasetFilepath'. Must use svmlib format for dataset.
    The randomSeed is used to use the same random position each run.
    'toDense = True' means that the data will be converted to a dense numpy array, which is easier to work with but can consume more memory. Set to False to keep it as a sparse matrix (scipy.sparse), which is more memory efficient for large datasets with many zeros.
    """
    if randomSeed:
        np.random.seed(randomSeed) # Set the seed internally of the function (optional)

    # Only do imports once we know the actual problem
    if problemName == "QDF":
        from QuadraticForm import QuadraticForm
        lossObj = QuadraticForm()
        initPos = np.random.uniform(-initialPosInterval, initialPosInterval, size=2)
    elif problemName == "Rosenbrock":
        from Rosenbrock import Rosenbrock
        lossObj = Rosenbrock(dim)
        initPos =  np.random.uniform(-initialPosInterval, initialPosInterval, dim)
    elif problemName == "LogReg":
        from LogisticRegression import LogisticRegression
        from DataLoader import loadDataAsNumpyArray
        X, y = loadDataAsNumpyArray(datasetFilepath, toDense = toDense, l2NormalizationOn=l2NormalizationOn)
        lossObj = LogisticRegression(data = [X,y], batchSize = batchSize)
        initPos = np.random.uniform(-initialPosInterval, initialPosInterval, lossObj.xDataLength)
    else:
        raise NotImplementedError(f"The problem '{problemName}' is not implemented.")
    return lossObj, initPos

