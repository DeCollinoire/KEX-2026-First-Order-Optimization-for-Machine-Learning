import numpy as np
from optimizers import sgd, momentum, nesterov, adam
from QuadraticForm import QuadraticForm
from testQdf import plot_path
import matplotlib.pyplot as plt

def testConvergence(optimizer, tol, nr_epochs):
    posHistory, lossHistory = optimizer(nr_epochs)
    posHistory = np.array(posHistory)
    
    fig = plt.figure(figsize=(8,6))
    fig.add_subplot(1,1,1)
    plot_path(optimizer.lossObj, posHistory, optimizer.__class__.__name__)
    plt.show()
    
    # Estimate errors as distance between two successive positions or from the minima (extrema) if known
    minima = optimizer.lossObj.minima()
    if minima is not None:
        errors = np.linalg.norm(posHistory - minima, axis=1)
    else:
        errors = np.linalg.norm(np.diff(posHistory, axis=0), axis=1)
    
    # Convergence ratios
    conv_ratios = errors[1:] / (errors[:-1] + 1e-15)
    
    # Find stopping point based on position change
    under_tol = np.where(errors < tol)[0]
    n_steps = under_tol[0] + 1 if under_tol.size > 0 else nr_epochs

    # Estimate convergence order q
    q = estimate_order(errors[:n_steps]) if n_steps > 3 else None

    return conv_ratios, n_steps, q

def estimate_order(errors):
    # Filter out zeros or negative values to avoid NaN in log
    errors = errors[errors > 1e-15]
    if len(errors) < 3: return None
    
    # Calculate log of successive error ratios
    # log(e_{k+1}/e_k)
    log_ratios = np.log(errors[1:] / errors[:-1])
    
    # q = log(e_{k+1}/e_k) / log(e_k/e_{k-1})
    numerator = log_ratios[1:]
    denominator = log_ratios[:-1]
    
    # Avoid division by zero (happens if convergence is perfectly linear)
    q_estimates = numerator / (denominator + 1e-15)
    
    # Return average of recent stable estimates, excluding Infs/NaNs
    valid_q = q_estimates[np.isfinite(q_estimates)]
    return np.mean(valid_q[-10:]) if valid_q.size > 0 else None

def main():
    lossObj = QuadraticForm()
    initPos = [2,5]

    optSGD = sgd.SGD(lossObj, initPos, lr = 0.1)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr = 0.1, decayFactor=0.3)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate = 0.1, decayFactor=0.3)
    optAdam = adam.Adam(lossObj, initPos, learningRate = 0.1, forgettingFactorM=0.999, forgettingFactorR=0.999)
    for optimizer in [optSGD, optNesterov, optMomentum, optAdam]:
        conv_ratios, N_steps, q = testConvergence(optimizer, tol = 1e-4, nr_epochs = 100)

        # Present
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Final position: {optimizer.pos}")
        print(f"Convergence ratios: {conv_ratios}")
        print(f"Number of steps to reach tolerance: {N_steps}")
        print(f"Estimated convergence order: {q}")

def main_alt():
    # NOTE: DOES NOT WORK YET
    # Config
    nr_epochs = 100
    lossObj = QuadraticForm()
    minima = lossObj.minima()
    initPos = [2,5]

    # Setup of optimizers
    optSGD = sgd.SGD(lossObj, initPos, lr = 0.1)
    optNesterov = nesterov.Nesterov(lossObj, initPos, lr = 0.1, decayFactor=0.3)
    optMomentum = momentum.Momentum(lossObj, initPos, learningRate = 0.1, decayFactor=0.3)
    optAdam = adam.Adam(lossObj, initPos, learningRate = 0.1, forgettingFactorM=0.999, forgettingFactorR=0.999)
    
    # Start testing
    errors = []
    for epoch in range(1, nr_epochs+1):
        for optimizer in [optSGD, optNesterov, optMomentum, optAdam]:
            optimizer.step()
            
            # Check convergence
            error = np.linalg.norm(optimizer.pos - minima)
            errors.append(error)
            
            if error < 1e-4:
                print(f"{optimizer.__class__.__name__} converged in {epoch} steps.")
                N_steps = epoch

                # Present
                print(f"Optimizer: {optimizer.__class__.__name__}")
                print(f"Final position: {optimizer.pos}")
                print(f"Convergence ratios: {conv_ratios}")
                print(f"Number of steps to reach tolerance: {N_steps}")
                print(f"Estimated convergence order: {q}")
 

if __name__=="__main__":
    main()

