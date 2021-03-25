import numpy as np
from itertools import product

from preprocessing import scale_list
from model import get_accuracy


def get_param_list(param_grid):
    """
    This function computes all possible combinations of hyperparameters for gridsearch.

    :param param_grid: dict of list
        Dict of list with all values to test for each hyperparameters

    :return: list of dict
        List of Dict with all all possible combinations of hyperparameters for gridsearch
    """
    param_list = []
    keys = param_grid.keys()
    values = param_grid.values()
    for param_value in product(*values):
        param_list.append(dict(zip(keys, param_value)))
    return param_list


def cross_validation(estimator, param_dict, X, y, k_list, n_folds, scale):
    """
    This function performs cross validation for a given estimator and a set of hyperparameters.

    :param estimator: estimator class
    :param param_dict: dict
    :param X: array
    :param y: array
    :param k_list: list of strings
    :param n_folds: int
    :param scale: list of booleans

    :return acc: float
        Mean accuracy computed with cross validation.
    """
    n = X[0].shape[0]
    fold_size = np.int(n/n_folds)
    idx = np.arange(n)
    np.random.shuffle(idx)
    X_shuffled, y_shuffled = [x[idx] for x in X], y[idx]
    average_acc = 0

    for i in range(n_folds):
        # Split data between train and valid
        idx_val_start = i * fold_size
        idx_val_end = (i + 1) * fold_size
        idx_val = np.arange(idx_val_start, idx_val_end)
        idx_train = np.delete(np.arange(n), idx_val)
        X_train, X_val = [x[idx_train] for x in X_shuffled], [x[idx_val] for x in X_shuffled]
        y_train, y_val = y_shuffled[idx_train], y_shuffled[idx_val]

        # scale data
        X_train, X_val = scale_list(X_train, X_val, scale)

        # Initiate and fit classifier
        clf = estimator(k_list, **param_dict)
        clf.fit(X_train, y_train)

        # Predict and evaluate on validation set
        acc_score = get_accuracy(clf, X_val, y_val)
        average_acc += acc_score

    acc = average_acc / n_folds
    return acc


def grid_search_cv(estimator, param_grid, X, y, k_list, n_folds, scale):
    """
    This functions performs gridsearch to find the best set of hyperparameters.

    :param estimator: estimator class
    :param param_grid: dict
    :param X: array
    :param y: array
    :param k_list: list of strings
    :param n_folds: int
    :param scale: list of booleans

    :return: float, dict
        Return the best performance and the best hyperparmaters.
    """
    param_list = get_param_list(param_grid)
    best_score = 0
    best_param = None
    for param_dict in param_list:
        acc = cross_validation(estimator, param_dict, X, y, k_list, n_folds, scale)
        if acc > best_score:
            best_score = acc
            best_param = param_dict.copy()
    return best_score, best_param
