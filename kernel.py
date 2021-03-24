import numpy as np
from scipy.spatial import distance


def kernel_function(kernel_name, X1, X2, params={}):
    if kernel_name == "linear":
        return X1@X2.T
    elif kernel_name == "gaussian":
        gamma = params["gamma"]
        quad_term = distance.cdist(X1, X2, "sqeuclidean")
        return np.exp(-gamma * quad_term)
    else:
        raise NameError("Kernel name not recognized")
