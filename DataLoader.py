import time
import timeit
import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

def loadDataAsNumpyArray(filepath, toDense = True):
    X, y = load_svmlight_file(filepath) # type: ignore - VS Code says it expects 3 values but running says 'expects 2 values'
    X = X.todense() if toDense else X
    y = np.array(y)
    return X, y

if __name__=="__main__":
    # How a csr/sp matrix work
    d = np.array([3, 4, 5, 7, 2, 6])     # data
    r = np.array([0, 0, 1, 1, 3, 3])     # rows
    c = np.array([2, 4, 2, 3, 1, 2])     # cols

    csr = csr_matrix((d, (r, c)), shape=(4, 5))
    print(csr.toarray())

    # Test sparse vs dense
    # 1. Load the dataset
    # RCV1 is naturally sparse (most values are zero) 
    X_sparse, y = load_svmlight_file("datasets/rcv1_train.binary") # type: ignore - VS Code says it expects 3 values but running says 'expects 2 values'

    # 2. Setup the Subsets
    n_samples = -1
    # X_subset shape: (1000, 47236)
    X_subset_sparse = X_sparse[:n_samples] # type: ignore
    X_subset_dense = X_subset_sparse.toarray()

    # 3. Create a weight vector that matches the number of features
    # Dimension must be (47236,) to align with (1000, 47236)
    n_features = X_subset_sparse.shape[1]
    weights = np.random.rand(n_features)

    # --- Benchmarking ---
    # Sparse Multiplication (CSR)
    start_sparse = timeit.default_timer()
    result_sparse = X_subset_sparse.dot(weights)
    end_sparse = timeit.default_timer()
    time_sparse = end_sparse - start_sparse

    # Dense Multiplication (NumPy)
    start_dense = timeit.default_timer()
    result_dense = np.dot(X_subset_dense, weights) # X @ weight
    end_dense = timeit.default_timer()
    time_dense = end_dense - start_dense

    # 4. Output Results
    print(f"Matrix Shape: {X_subset_sparse.shape}")
    print(f"Number of Non-Zeros: {X_subset_sparse.nnz}")
    print("-" * 30)
    print(f"Time Sparse (CSR): {time_sparse:.6f} s")
    print(f"Time Dense:        {time_dense:.6f} s")
    print(f"Speedup Factor:    {time_dense / time_sparse:.2f}x")

    # Verification of correctness
    assert np.allclose(result_sparse, result_dense), "Results do not match!"

    # Test loadDataAsNumpyArray
    X, y = loadDataAsNumpyArray("datasets/australian_scale") # type: ignore - VS Code says it expects 3 values but running says 'expects 2 values'    

    print(X[:10])
    print(y[:10])
