
from copy import deepcopy
from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import expit

import sklearn.preprocessing
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.ticker as ticker


def logRegLoss(weights, X, y):
    # return (1 / len(y)) * np.sum(np.log(1 + np.exp(-y * (X.dot(weights)))))
    return np.sum(np.logaddexp(0,
                               -y * (X @
                                     weights)))#/len(y)

def logRegGrad(weights, X, y):
    # return -(X.T @ (y * expit(-(y * (X @ weights))))) / len(y) #expit is a sigmoid function "The expit function, also known as the logistic sigmoid function, is defined as expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function."
    return -(X.T.dot((y * expit(-(y * (X.dot(weights)))))) )#/ len(y))

    #Without divided by samples on grad and loss currently.

    ####
    #Eplicit form

    #"The expit function, also known as the logistic sigmoid function, is defined as expit(x) = 1/(1+exp(-x))
    # Amount of samples m
    # m = len(y)
    # result = 0
    # z = 0
    # for i in range(m):
    #     z = y[i] * (X[i].dot(weights))
    #     result = result + expit(-z) * (-y[i] * X[i])
    # result = result #/ m
    # return result
    ####

# def logRegLoss(weights, X, y):
#     z = y * X.dot(weights)
#     # return (1 / self.xDataListLength) * np.sum(np.log(1 + np.exp(-self.y * (self.X @ position))))
#     # This is equivalent to log(1 + exp(-z)) but stable
#     return np.mean(np.logaddexp(0, -z))

def sgdTrain(lrList, X, y, nrEpochs):
    # Initialize plot figure
    plt.figure(figsize=(10, 8))

    # Train
    for lr in lrList:
        historyLoss = []

        weights = np.zeros(X.shape[1])

        for i in range(nrEpochs):
            historyLoss.append(logRegLoss(weights, X, y))

            # Step with gradient descent
            weights = weights - (lr * logRegGrad(weights=weights, X=X, y=y))

        plt.plot(historyLoss, marker="o", label=f"lr={lr}")

    # Plot

    # --- Grid Customization Start ---
    ax = plt.gca()  # Get current axes

    # Set grid lines every 10 epochs (X-axis)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # Set grid lines every 2000 units of loss (Y-axis)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    # --- Grid Customization End ---

    plt.title("GD fullbatch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("linear")
    plt.grid(True)
    plt.legend()
    plt.grid(True)

    plt.show()

def main():

    # Load dataset
    X, y = load_svmlight_file("rcv1_train.binary")

    #Normalization
    # scaler = MaxAbsScaler()
    # X = scaler.fit_transform(X)

    #X = normalize(X, norm='l2', axis=1)

    # Setup optimizer
    weights = np.zeros(X.shape[1]) #weights = [0 for _ in range((len(X[0])))]
    lr = 0.1


    # Train
    nrEpochs = 100
    historyLoss = []

    for i in range(nrEpochs):
        historyLoss.append(logRegLoss(weights, X, y))

        # Step with gradient descent
        weights = weights - (lr * logRegGrad(weights=weights, X=X, y=y))


    # Plot

    plt.figure(figsize=(10, 8))
    plt.plot(historyLoss, marker="o", label="Loss")
    plt.title("GD fullbatch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("linear")
    plt.grid(True)
    plt.legend()
    plt.grid(True)

    plt.show()


    ####
    sgdTrain(lrList=[0.001, 0.01, 0.05, 0.1], X=X, y=y, nrEpochs=nrEpochs)

if __name__ == "__main__":
    main()

