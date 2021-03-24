import numpy as np
from itertools import product


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.keep_cols = []

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.keep_cols = np.argwhere(self.std > 0).reshape(-1)
        self.mean = self.mean[self.keep_cols]
        self.std = self.std[self.keep_cols]

    def transform(self, X):
        X = X[:, self.keep_cols]
        X = X - self.mean
        X = X / self.std
        return X


def scale_list(X_tr_l, X_te_l, scale_l):
    X_scaled_tr, X_scaled_te = [], []
    for k in range(len(X_tr_l)):
        if scale_l[k]:
            scaler = Scaler()
            scaler.fit(X_tr_l[k])
            X_scaled_tr.append(scaler.transform(X_tr_l[k]))
            X_scaled_te.append(scaler.transform(X_te_l[k]))
        else:
            X_scaled_tr.append(X_tr_l[k])
            X_scaled_te.append(X_te_l[k])
    return X_scaled_tr, X_scaled_te


def get_idx_pattern(k):
    # Define all possible substrings
    characters = ["A", "T", "C", "G"]
    cart_prod = k * [characters]
    idx = np.array(list((product(*cart_prod))))
    idx = np.apply_along_axis(lambda u: ''.join(u), 1, idx)
    return idx


def spectrum_transformation(x, k, idx):
    transformed_x = np.zeros(len(idx))
    for i in range(len(x) - (k - 1)):
        idx_match = np.argwhere(idx == x[i:i+k])
        transformed_x[idx_match] += 1
    return transformed_x


def spectrum_phi(X, k):
    idx = get_idx_pattern(k)
    # Apply transformation to dataset
    lambda_function = lambda x: spectrum_transformation(x, k, idx)
    transformed_X = np.array(list(map(lambda_function, X)))
    return transformed_X


def test_spectrum_phi():
    X = np.array(['AATT', 'ATAT'])
    k = 2
    true_output = np.array([
        [1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 2., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    assert (true_output == spectrum_phi(X, k)).all(), \
        'Problem with spectrum phi, results obtained = {}'.format(spectrum_phi(X, k))


def get_matching_array(idx, m):
    # convert idx to replace strings with arrays of single characters
    idx_t = np.array(list(map(lambda x: np.array(list(x)), idx)))
    matchings = []
    for i, pattern in enumerate(idx_t):
        idx_match = np.argwhere((pattern != idx_t).sum(axis=1) <= m)
        idx_match = idx_match[idx_match != i]
        matchings.append(idx_match)
    return matchings


def mismatch(X, k, m):
    assert m <= k/2, "k = {} and m = {} , reduce the number of mismatchs".format(k, m)
    idx = get_idx_pattern(k)
    # create matching arrays
    matching_arr = get_matching_array(idx, m)
    final_X = X.copy()
    for i, pattern in enumerate(idx):
        final_X[:, matching_arr[i]] += X[:, i].reshape((-1, 1))
    return final_X


def test_mismatch():
    X = np.array(['AATT', 'ATAT'])
    k = 2
    m = 1
    true_output = np.array([
        [2., 3., 2., 2., 2., 2., 1., 1., 1., 2., 0., 0., 1., 2., 0., 0.],
        [3., 2., 2., 2., 1., 3., 1., 1., 1., 2., 0., 0., 1., 2., 0., 0.]
    ])
    X = spectrum_phi(X, k)
    assert (true_output == mismatch(X, k, m)).all(), \
        'Problem with mismatch, results: {}'.format(mismatch(X, k, m))


def main():
    test_spectrum_phi()
    test_mismatch()


if __name__ == '__main__':
    main()
