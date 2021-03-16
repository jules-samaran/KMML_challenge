import os
import yaml
from easydict import EasyDict as edict

import numpy as np
import pandas as pd
from datetime import datetime
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


def load_data(idx, split, type):
    if type == "raw":
        suffix = ""
    else:
        suffix = "_mat100"
    x_filename = "X{}{}{}.csv".format(split, idx, suffix)
    x_path = os.path.join(os.getcwd(), "data", x_filename)
    x_df = pd.read(x_path)
    x = np.array(x_df.iloc[:, 1].values)
    if split == "tr":
        y_filename = "Ytr{}.csv".format(idx)
        y_path = os.path.join(os.getcwd(), "data", y_filename)
        y_df = pd.read(y_path)
        y = np.array(y_df.iloc[:, 1].values)
        return x, y


def get_pred_subms(model, type):
    predictions = []
    for idx in range(3):
        X_tr, y_tr = load_data(idx, "tr", type)
        X_te = load_data(idx, "te", type)
        model.fit(X_tr, y_tr)
        predictions.append(model.predict(X_te))
    pred_df = pd.Dataframe(np.concatenate(predictions, axis=0), columns=["Id", "Bound"])
    assert pred_df.shape == (3000, 2)
    return pred_df


def subs_wrapper(cfg_path, subs_dir):
    # load config
    with open(cfg_path) as f:
        cfg = edict(yaml.load(f))

    # get hyperparams from CV
    model = None # initialize model
    type = cfg.DATA.TYPE

    date = datetime.now().strftime("%d_%H:%M:")
    dir_name = "{}_{}_{}".format(date, model.name, model.k_name)
    save_dir = os.path.join(subs_dir, dir_name)
    os.mkdir(save_dir)

    preds_df = get_pred_subms(model, type)
    preds_df.to_csv(os.path.join(save_dir, "predictions.csv"))
    cfg_name = os.path.split(cfg_path)[-1]
    print("Done with {}.".format(cfg_name))
