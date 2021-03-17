import os
import shutil
import yaml
from easydict import EasyDict as edict

import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

from model import models_dic


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
        X_train, X_val = X_shuffled[~idx_val], X_shuffled[idx_val]
        y_train, y_val = y_shuffled[~idx_val], y_shuffled[idx_val]

        # Initiate and fit classifier
        clf = estimator(**param_dict)
        clf.fit(X_train, y_train)

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
    x_df = pd.read_csv(x_path, header=None, sep=" ")
    x = np.array(x_df.values)
    if split == "tr":
        y_filename = "Ytr{}.csv".format(idx)
        y_path = os.path.join(os.getcwd(), "data", y_filename)
        y_df = pd.read_csv(y_path)
        y = np.array(y_df.iloc[:, 1].values)
        y = np.where(y == 1, 1, -1)
        return x, y
    return x


def get_pred_subms(cfg, type):
    predictions = []
    full_log = ""

    for idx in range(3):
        X_tr, y_tr = load_data(idx, "tr", type)
        X_te = load_data(idx, "te", type)

        # get hyperparams from CV
        model_class = models_dic[cfg.MODEL_NAME]
        best_val_score, best_param = grid_search_cv(model_class, cfg.grid_hparams, X_tr, y_tr, cfg.N_FOLDS)
        type = cfg.DATA.type

        model = model_class(**best_param)
        model.fit(X_tr, y_tr)
        train_score = get_accuracy(model, X_tr, y_tr)
        predictions.append(model.predict(X_te))
        log = "Split {}: train_acc: {:.3f}, val_acc: {:.3f}, selected parameters: {} \n".format(idx, best_val_score,
                                                                                           train_score, best_param)
        full_log += log
        print("Done Split {}".format(idx))

    pred_array = np.concatenate(predictions, axis=0)
    # For evaluation reconvert to 0 and 1
    pred_array = np.where(pred_array == 1, 1, 0)
    pred_df = pd.DataFrame(np.concatenate((np.arange(pred_array.shape[0]).reshape((-1, 1)),
                                           pred_array.reshape((-1, 1))), axis=1),
                           columns=["Id", "Bound"])
    assert pred_df.shape == (3000, 2)
    return pred_df, full_log


def subs_wrapper(cfg_path, subs_dir):
    # load config
    with open(cfg_path) as f:
        cfg = edict(yaml.load(f))

    date = datetime.now().strftime("%d_%H:%M:%S:")
    dir_name = "{}_{}".format(date, cfg.MODEL_NAME)
    save_dir = os.path.join(subs_dir, dir_name)

    preds_df, result_log = get_pred_subms(cfg, type)

    os.mkdir(save_dir)
    shutil.copy(cfg_path, save_dir)
    preds_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    with open(os.path.join(save_dir, "log.txt"), "w") as text_file:
        text_file.write(result_log)
    cfg_name = os.path.split(cfg_path)[-1]
    print("Done with {}.".format(cfg_name))
