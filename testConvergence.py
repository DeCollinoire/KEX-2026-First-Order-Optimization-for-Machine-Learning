import numpy as np
from optimizers import sgd, momentum, nesterov, adam
from optimizers.loss.loss import QuadraticForm

def testConvergence(optimizer, lossFunc, tol, nr_epochs):
    """
    Function for testing the convergence of an optimizer function (optFunc) on a certain loss function (lossFunc).
    It measures:
        1. A history of all convergence ratios.
        2. A convergence constant given more than four steps.
        3. The number of steps to reach a certain tolerance.
    """
    # Run optimization
    posHistory = []
    lossHistory = []
    
    # Convergence ratio = loss[i] / loss[i-1]
    conv_ratios = lossHistory / np.roll(lossHistory, -1)
    
    # Convergence steps = First index where loss < tol
    N_steps = int(np.argmin(lossHistory < tol))

    return conv_ratios, N_steps

def main():
    optSGD = sgd.SGD([0, 0], lr = 0.1)
    optNesterov = nesterov.Nesterov([0,0], lr = 0.1)
    for optimizer in [optSGD, optNesterov]:
        testConvergence(optimizer, QuadraticForm, tol = 1e-2, nr_epochs = 100)

if __name__=="__main__":
    main()

