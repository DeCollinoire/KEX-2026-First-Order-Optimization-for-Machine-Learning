import numpy as np
from optimizers.optimizer import SGD
from optimizers.loss import QuadraticForm

def testConvergence(optimizer, lossFunc, tol, nr_epochs):
    """
    Function for testing the convergence of an optimizer function (optFunc) on a certain loss function (lossFunc).
    It measures:
        1. A history of all convergence ratios.
        2. A convergence constant given more than four steps.
        3. The number of steps to reach a certain tolerance.
    """
    # Run optimization
    paramsHistory = []
    lossHistory = []
        
    # Convergence ratio = loss[i] / loss[i-1]
    conv_ratios = lossHistory / np.roll(lossHistory, -1)
    
    # Convergence steps = First index where loss < tol
    N_steps = int(np.argmin(lossHistory < tol))

    return conv_ratios, N_steps

def main():
    for optimizerClass in [SGD]:
        optimizer = optimizerClass([0, 0], lr = 0.1, gamma = 0.1, batch_size = 100, )
        testConvergence(optimizer, QuadraticForm, 1e-2, nr_epochs=100)

if __name__=="__main__":
    main()

