
from copy import deepcopy
from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import expit

import sklearn.preprocessing
from sklearn.preprocessing import MaxAbsScaler


# def logRegLoss(weights, X, y):
#     return (1 / len(y)) * np.sum(np.log(1 + np.exp(-y * (X.dot(weights)))))

def logRegGrad(weights, X, y):
    # return -(X.T @ (y * expit(-(y * (X @ weights))))) / len(y) #expit is a sigmoid function
    return -(X.T.dot((y * expit(-(y * (X.dot(weights)))))) / len(y))

def logRegLoss(weights, X, y):
    z = y * X.dot(weights)
    # This is equivalent to log(1 + exp(-z)) but stable
    return np.mean(np.logaddexp(0, -z))



def main():

    # Load dataset
    X, y = load_svmlight_file("rcv1_train.binary")

    #Normalization
    #scaler = MaxAbsScaler()
    #X = scaler.fit_transform(X)

    #X = normalize(X, norm='l2', axis=1)

    # Setup optimizer
    weights = np.zeros(X.shape[1]) #weights = [0 for _ in range((len(X[0])))]
    lr = 20


    # Train
    nrEpochs = 50
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
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()
