import numpy as np
import scipy as sp
from itertools import product


def spectrum_transformation(x, k, idx):
    transformed_x = np.zeros(len(idx))
    for i in range(len(x) - (k - 1)):
        idx_match = np.argwhere(idx == x[i:i+k])
        transformed_x[idx_match] += 1
    return transformed_x


def spectrum_phi(X, k):
    # Define all possible substrings
    characters = ["A", "T", "C", "G"]
    cart_prod = k * [characters]
    idx = np.array(list((product(*cart_prod))))
    idx = np.apply_along_axis(lambda u: ''.join(u), 1, idx)

    # Apply transformation to dataset
    lambda_function = lambda x: spectrum_transformation(x[0], k, idx)
    transformed_X = np.apply_along_axis(lambda_function, 1, X)
    return transformed_X


def test_spectrum_phi():
    X = np.array([['AATT'], ['ATAT']])
    k = 2
    true_output = np.array([
        [1. ,1. ,0. ,0. ,0. ,1. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
        [0. ,2. ,0. ,0. ,1. ,0. ,0. ,0. ,0. ,0. ,0., 0., 0., 0., 0., 0.]
    ])
    assert (true_output == spectrum_phi(X, 2)).all(), 'Problem with spectrum phi'


def main():
    test_spectrum_phi()


if __name__ == '__main__':
    main()