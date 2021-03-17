import numpy as np
from scipy.spatial import distance


def kernel_function(kernel_name, X1, X2, params={}):
    if kernel_name == "linear":
        return X1@X2.T
    elif kernel_name == "gaussian":
        s = params["sigma"]
        quad_term = distance.cdist(X1, X2, "sqeuclidean")
        return np.exp(-(0.5 * 1/s**2) * quad_term)
    else:
        raise NameError("Kernel name not recognized")
