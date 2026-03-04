import numpy as np
import matplotlib.pyplot as plt
from optimizers.optimizer import Optimizer
from optimizers.sgd import SGD
from QuadraticForm import QuadraticForm

def test_hyperparameter_sensitivity(optimizer: Optimizer, hyperparams: dict):
    adj_factors = [0.5, 0.75, 1, 1.25, 1.5]
    results = []
    for h_name, baseval in hyperparams.items():
        for factor in adj_factors:
            # Adjust the basevalue in the optimizer
            setattr(optimizer, h_name, baseval * factor)
            
            # Run the optimization and get the results
            posHistory, lossHistory = optimizer(nr_epochs=100)
        N_steps = int(np.argmin(np.greater(lossHistory, 0.1)))
    return results

if __name__ == "__main__":    
    A = np.array([
        [19, 0],
        [0, 5]
        ])
    b = np.array([1, 5])
    lossObj = QuadraticForm(A, b)
    initPos = np.array([1.0, 1.0, 1.0])
    
    # Test learning rate
    h = {
        "lr": 0.1,
    }
    optimizer = SGD(lossObj, initPos, **h) # lr = 0.1

    results = test_hyperparameter_sensitivity(optimizer, h)
    print(results)
