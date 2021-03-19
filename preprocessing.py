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
    idx = np.array(list(map(lambda u: ''.join(u), idx)))

    lambda_function = lambda x: spectrum_transformation(x, k, idx)
    transformed_X = np.array(list(map(lambda_function, X)))
    return transformed_X


def test_spectrum_phi():
    X = np.array(['AATT', 'AATG'])
    k = 2
    true_output = np.array([
        [1. ,1. ,0. ,0. ,0. ,1. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
        [0. ,2. ,0. ,0. ,1. ,0. ,0. ,0. ,0. ,0. ,0., 0., 0., 0., 0., 0.]
    ])
    assert (true_output == spectrum_phi(['AATT', 'ATAT'], 2)).all(), 'Problem with spectrum phi'


def main():
    test_spectrum_phi()


if __name__ == '__main__':
    main()