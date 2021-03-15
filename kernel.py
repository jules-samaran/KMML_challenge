import numpy as np


def kernel_function(kernel_name, X1, X2):
    if kernel_name == "linear":
        return X1@X2.T
