import numpy as np
from optimizers import sgd, momentum, nesterov, adam
from QuadraticForm import QuadraticForm

def testConvergence(optimizer, tol, nr_epochs):
    posHistory, lossHistory = optimizer(nr_epochs)
    
    # Convergence ratios: L(t+1) / L(t)
    conv_ratios = lossHistory[1:] / (lossHistory[:-1] + 1e-15) # Avoid div by zero
    
    # Find first index where loss < tol
    under_tol = np.where(lossHistory < tol)[0]
    n_steps = under_tol[0] if under_tol.size > 0 else nr_epochs

    # Estimate convergence order q (if enough points)
    q = estimate_order(lossHistory[:n_steps]) if n_steps > 2 else None

    return conv_ratios, n_steps, q

def estimate_order(lossHistory):
    # Need three points to estimate convergence order q
    numerator = np.log(lossHistory[2:] / lossHistory[1:-1])
    denominator = np.log(lossHistory[1:-1] / lossHistory[:-2])
    q_estimates = numerator / (denominator + 1e-15)
    
    # Return the average of the last few stable steps
    return np.mean(q_estimates[-10:])

def main():
    qdf = QuadraticForm()
    initPos = [0,0]

    optSGD = sgd.SGD(qdf, initPos, lr = 0.1)
    optNesterov = nesterov.Nesterov(qdf, initPos, lr = 0.1)
    for optimizer in [optSGD, optNesterov]:
        conv_ratios, N_steps, q = testConvergence(optimizer, tol = 1e-2, nr_epochs = 100)
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Convergence ratios: {conv_ratios}")
        print(f"Number of steps to reach tolerance: {N_steps}")
        print(f"Estimated convergence order: {q}")

if __name__=="__main__":
    main()

