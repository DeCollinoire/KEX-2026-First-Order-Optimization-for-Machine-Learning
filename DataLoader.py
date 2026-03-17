import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

def loadDataAsNumpyArray(filepath):
    X, y = load_svmlight_file("datasets/australian_scale") # type: ignore - VS Code says it expects 3 values but running says 'expects 2 values'
    X = X.todense()
    y = np.array(y)
    return X, y


if __name__=="__main__":
    # How a csr/sp matrix work
    d = np.array([3, 4, 5, 7, 2, 6])     # data
    r = np.array([0, 0, 1, 1, 3, 3])     # rows
    c = np.array([2, 4, 2, 3, 1, 2])     # cols

    csr = csr_matrix((d, (r, c)), shape=(4, 5))
    print(csr.toarray())

    # Load data
    X, y = loadDataAsNumpyArray("datasets/australian_scale") # type: ignore - VS Code says it expects 3 values but running says 'expects 2 values'
    print(X[:10])
    print(y[:10])
