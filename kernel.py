import numpy as np
from scipy.spatial import distance


def kernel_function(kernel_name, X1, X2, params={}):
    """
    This function computes kernel matrices.

    :param kernel_name: string
    :param X1: array
    :param X2: array
    :param params: dict
        Hyperparameters of the kernel

    :return: array
        Gram matrix of the kernel
    """
    if kernel_name == "linear":
        return X1@X2.T
    elif kernel_name == "gaussian":
        gamma = params["gamma"]
        quad_term = distance.cdist(X1, X2, "sqeuclidean")
        return np.exp(-gamma * quad_term)
    else:
        raise NameError("Kernel name not recognized")


def kernel_wrapper(k_names, X1_l, X2_l, params=[]):
    """
    This function is a wrapper to handle sum kernels.
    A single kernel is considered as a sum of 1 element.

    :param k_names: list of string
        List of kernel names to use
    :param X1_l: array
    :param X2_l: array
    :param params: list of dict
        List of hyperparameters dict for kernels

    :return: array
        Gram matrix of the sum kernel.
    """
    K_list = []
    for i in range(len(k_names)):
        K_list.append(kernel_function(k_names[i], X1_l[i], X2_l[i], params[i]))
    sum_K = np.sum(K_list, axis=0)
    return sum_K
