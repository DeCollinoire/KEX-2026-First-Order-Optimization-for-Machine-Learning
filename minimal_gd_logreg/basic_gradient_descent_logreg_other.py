
from copy import deepcopy
from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import expit

class GradientDescent:
    def __init__(self, lr):
        self.lr = lr
        #self.weights = weights

    def step(self, weights, X, y):
        weights = weights - (self.lr * logRegGrad(weights=weights, X=X, y=y))
        return weights


def logRegLoss(weights, X, y):
    return (1/len(y)) * np.sum(np.log(1 + np.exp(-y * (X @ weights))))

def logRegGrad(weights, X, y):
    return -(X.T @ (y * expit(-(y * (X @ weights))))) / len(y) #expit is a sigmoid function



def main():

    # Load dataset
    X, y = load_svmlight_file("rcv1_train.binary")

    # Setup optimizer
    weights = np.zeros(X.shape[1]) #weights = [0 for _ in range((len(X[0])))]
    lr = 1800

    optimizer = GradientDescent(lr=lr)

    # Train
    nrEpochs = 50
    historyLoss = []

    for i in range(50):
        historyLoss.append(logRegLoss(weights, X, y))
        weights = optimizer.step(weights=weights, X=X, y=y)


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