import numpy as np
from itertools import product

def get_param_list(param_grid):
    param_list = []
    keys = param_grid.keys()
    values = param_grid.values()
    for param_value in product(*values):
        param_list.append(dict(zip(keys, param_value)))
    return param_list


def get_accuracy(classifier, X, y):
    y_pred = classifier.predict(X)
    acc = np.mean(y_pred == y)
    return acc


def cross_validation(estimator, param_dict, X, y, n_folds):
    n = X.shape[0]
    fold_size = np.int(n/n_folds)
    idx = np.arange(n)
    np.random.shuffle(idx)
    X_shuffled, y_shuffled = X[idx], y[idx]
    average_acc = 0

    for i in range(n_folds):
        # Split data between train and valid
        idx_val_start = i * fold_size
        idx_val_end = (i + 1) * fold_size
        idx_val = np.arange(idx_val_start, idx_val_end)
        X_train, X_val = X[~idx_val], X[idx_val]
        y_train, y_val = y[~idx_val], y[idx_val]

        # Initiate and fit classifier
        clf = estimator(**param_dict)
        clf.fit(X, y)

        # Predict and evaluate on validation set
        acc_score = get_accuracy(clf, X_val, y_val)
        average_acc += acc_score

    return average_acc / n_folds


def grid_search_cv(estimator, param_grid, X, y, n_folds):
    param_list = get_param_list(param_grid)
    best_score = 0
    best_param = None
    for param_dict in param_list:
        acc = cross_validation(estimator, param_dict, X, y, n_folds)
        if acc > best_score:
            best_score = acc
            best_param = param_dict.copy()
    return best_score, best_param

