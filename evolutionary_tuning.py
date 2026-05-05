import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

# Project files
from utils import plotHistoryGraph, train, train_external_batching, setupProblem
from DataLoader import loadDataAsNumpyArray
from LogisticRegression import LogisticRegression
from Rosenbrock import Rosenbrock
from optimizers import sgd, nesterov, momentum, adam

"""
NOTE: This version implements a gridsearch approach to sensitivity, i.e. it tests all combinations of the given hyperparams.
All optimizer variants are created first, as opposed to modifying them one-by-one during testing.
"""

class LogisticRegression_ExternalBatching():
    """ 
    This version of logistic regression externalizes batching from the lossObj, which allows us to use the same lossObj for all optimizers and batch sizes in the sensitivity test.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def setCurrentBatch(self, X_batch, y_batch):
        self.X_batch = X_batch
        self.y_batch = y_batch

    def getCurrentBatch(self):
        return self.X_batch, self.y_batch

    def evaluate_loss(self, weights):
        nrOfSamples = self.X.shape[0]
        return np.sum(np.log(1 + np.exp(-self.y * (self.X @ weights)))) / nrOfSamples

    def evaluate_gradient(self, weights):
        X, y = self.getCurrentBatch()
        return -(X.T @ (y * expit(-(y * (X @ weights))))) # expit is a sigmoid function

def createVariants(lossObj, initPos):
    variants = {"SGD": [], "Nesterov": [], "Momentum": [], "Adam": []}

    # SGD
    for lr in [0.01, 0.1, 0.2, 0.5, 1, 5, 10, 100, 1000, 2000]:
        variants["SGD"].append(sgd.SGD(lossObj, initPos, lr=lr))

    # Nesterov
    for lr in [0.01, 0.1, 0.2, 0.5, 1, 5, 10]:
        for decayFactor in [0.5, 0.7, 0.9, 0.99]:
            variants["Nesterov"].append(nesterov.Nesterov(lossObj, initPos, lr=lr, decayFactor=decayFactor))
    
    # Momentum
    for lr in [0.01, 0.1, 0.2, 0.5, 1, 5, 10]:
        for decayFactor in [0.5, 0.7, 0.9, 0.99]:
            variants["Momentum"].append(momentum.Momentum(lossObj, initPos, learningRate=lr, decayFactor=decayFactor))

    # Adam
    for lr in [0.01, 0.1, 0.2, 0.5, 0.75]:
        for forgettingFactorM in [0.7, 0.9, 0.99]:
            for forgettingFactorR in [0.7, 0.9, 0.99]:
                variants["Adam"].append(adam.Adam(lossObj, initPos, learningRate=lr, forgettingFactorM=forgettingFactorM, forgettingFactorR=forgettingFactorR))

    return variants


def evolutionaryTest(optimizerList, lossObj, X, y, nrEpochs=20, batchSize=1000, n_generations=5):
    """
    Interactive evolutionary hyperparameter search.
    Each generation: train all → plot → user selects survivors → mutate into next gen.
    """
    import copy, json

    population = optimizerList
    sigma = 0.3          # log-space mutation std for lr
    crossover_p = 0.5
    offspring_per_parent = 3
    elitism = True

    for gen in range(1, n_generations + 1):
        print(f"\n{'='*50}\n  Generation {gen}  (population: {len(population)})\n{'='*50}")

        # --- Train ---
        train_external_batching(population, lossObj, X, y,
                                nrEpochs=nrEpochs, batchSize=batchSize, printProgress=True)

        # --- Plot & selection ---
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, opt in enumerate(population):
            ax.plot(opt.lossHistory, label=f"[{i}] {opt.__class__.__name__} {opt.getHyperparamStr()}", alpha=0.7)
        ax.set_title(f"Generation {gen} — enter survivor indices below")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(); ax.legend(fontsize=8)
        plt.tight_layout(); plt.show()

        # --- User picks survivors ---
        raw = input(f"\nEnter survivor indices (comma-separated, e.g. 0,2,5): ")
        survivor_ids = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        survivors = [population[i] for i in survivor_ids if i < len(population)]
        print(f"  Survivors: {[opt.__class__.__name__ for opt in survivors]}")

        if not survivors:
            print("No survivors selected — stopping.")
            break
        if gen == n_generations:
            print("Final generation reached.")
            break

        # --- Evolve: mutate + crossover ---
        next_pop = list(survivors) if elitism else []

        for parent in survivors:
            for _ in range(offspring_per_parent):
                child = copy.deepcopy(parent)
                hp = child.getHyperparams()  # dict: {"lr": ..., "decayFactor": ..., etc.}

                # Crossover: swap some hyperparams with a random other survivor
                if len(survivors) > 1 and np.random.rand() < crossover_p:
                    donor = np.random.choice([s for s in survivors if s is not parent])
                    donor_hp = donor.getHyperparams()
                    for k in hp:
                        if np.random.rand() < 0.5:
                            hp[k] = donor_hp[k]

                # Mutation: log-normal perturbation for lr, normal for others
                for k, v in hp.items():
                    if "lr" in k.lower() or "rate" in k.lower():
                        hp[k] = float(np.clip(v * np.exp(np.random.normal(0, sigma)), 1e-5, 10))
                    else:
                        hp[k] = float(np.clip(np.random.normal(v, sigma * 0.15), 0.05, 0.9999))

                child.setHyperparams(hp)
                child.reset()  # clear lossHistory and velocity/moments
                next_pop.append(child)

        print(f"  Next generation: {len(next_pop)} optimizers")
        population = next_pop

    return population  # final survivors

def main():
    # Config
    randomSeed = 25
    datasetFilepath = "datasets/rcv1_train.binary" # This is also used for plot titles
    initialPosInterval = 0.1
    batchSize = 1000

    # Setup
    print("Setting up...")
    np.random.seed(randomSeed)
    X, y = loadDataAsNumpyArray(datasetFilepath, toDense=False, l2NormalizationOn=False)
    nrSamples, nrFeatures = X.shape # type: ignore

    lossObj = LogisticRegression_ExternalBatching(X, y) # Use the external batching version of logistic regression for this test. It works fine to use the internal batching of the lossObj also.
    initPos = np.random.uniform(-initialPosInterval, initialPosInterval, nrFeatures) # Initialize position based on the number of features in the dataset

    # Create variants
    variants = createVariants(lossObj, initPos) # Creates all variants of sgd, nesterov, momentum and adam in a dictionary
    joinedOptList = [opt for optList in variants.values() for opt in optList] # Join all optimizers into one list for training
    print(f"Setup finished!")

    # Train
    print(f"\n--- Training ---")
    train_external_batching(joinedOptList, lossObj, X, y, nrEpochs=20, batchSize=batchSize, printProgress=True)
    print(f"Training finished.")

    # Plot
    for optClassName, optVariants in variants.items():
        plt.figure(str(optClassName+"_sensitivity_test"), figsize=(10, 6))
        for opt in optVariants:
            plotHistoryGraph(opt.lossHistory, title=f"{opt.__class__.__name__} Hyperparameter Sensitivity, lossObj = {lossObj.__class__.__name__}, problem = {datasetFilepath}", label=f"{opt.__class__.__name__}, {opt.getHyperparamStr()}", ylabel="Loss", yscale="linear")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
