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


def kernel_wrapper(k_names, X1_l, X2_l, params=[]):
    K_list = []
    for i in range(len(k_names)):
        K_list.append(kernel_function(k_names[i], X1_l[i], X2_l[i], params[i]))
    sum_K = np.sum(K_list, axis=0)
    return sum_K
