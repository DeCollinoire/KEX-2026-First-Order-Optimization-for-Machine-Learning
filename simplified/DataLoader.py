import time
import timeit
import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MaxAbsScaler

def loadDataAsNumpyArray(filepath, toDense = False, L2NormalizationOn=False):
    X, y = load_svmlight_file(filepath) # type: ignore - VS Code says it expects 3 values but running says 'expects 2 values'

    # Normalizing
    if L2NormalizationOn == True:
        scaler = MaxAbsScaler() # doesn't seem to make a difference from "normalize"
        X = scaler.fit_transform(X)

        # X = normalize(X, norm='l2', axis=1)

    X = np.asarray(X.todense()) if toDense else X
    return X, y

if __name__=="__main__":
    datasetFilepath = "datasets/rcv1_train.binary"
    X, y = loadDataAsNumpyArray(datasetFilepath, L2NormalizationOn=True)
    print(type(X))          # is it still sparse?
    print(X[0].sum())       # for L2 norm, row norms should all be ~1.0
    import numpy as np
    from sklearn.preprocessing import normalize
    # verify a few row norms
    dense_rows = np.asarray(X[:5].todense())
    print(np.linalg.norm(dense_rows, axis=1))  # should all be 1.0