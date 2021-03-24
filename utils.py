import os
import shutil
import yaml
from easydict import EasyDict as edict

import numpy as np
import scipy.sparse
import pandas as pd
from datetime import datetime
from itertools import product

from model import models_dic
from preprocessing import spectrum_phi, mismatch, Scaler, scale_list


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


def cross_validation(estimator, param_dict, X, y, k_list, n_folds, scale):
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

    return average_acc / n_folds


def grid_search_cv(estimator, param_grid, X, y, k_list, n_folds, scale):
    param_list = get_param_list(param_grid)
    best_score = 0
    best_param = None
    for param_dict in param_list:
        acc = cross_validation(estimator, param_dict, X, y, k_list, n_folds, scale)
        if acc > best_score:
            best_score = acc
            best_param = param_dict.copy()
    return best_score, best_param


def load_spectrum(split, idx, k):
    suffix = "spectrum{}".format(k)
    x_filename = "X{}{}{}.npy".format(split, idx, suffix)
    path_saved = os.path.join(os.getcwd(), "data", "processed", x_filename)
    if os.path.exists(path_saved):
        x = np.load(path_saved)
    else:
        x_path = os.path.join(os.getcwd(), "data", "original", "X{}{}.csv".format(split, idx))
        x_df = pd.read_csv(x_path, header=0)
        x = np.array(x_df.iloc[:, 1].values)
        x = spectrum_phi(x, k)
        np.save(path_saved[:-4], x)
    return x


def load_mat100(split, idx):
    suffix = "_mat100"

    x_filename = "X{}{}{}.csv".format(split, idx, suffix)
    x_path = os.path.join(os.getcwd(), "data", "original", x_filename)
    x_df = pd.read_csv(x_path, header=None, sep=" ")
    x = np.array(x_df.values)
    return x


def load_data(idx, split, type):
    if type.startswith("spectrum"):
        k = int(type[8:])
        x = load_spectrum(split, idx, k)
    elif type.startswith("mismatching"):
        k = int(type[13:])
        m = int(type[11])
        x = load_spectrum(split, idx, k)
        x = mismatch(x, k, m)
    elif type == "mat100":
        x = load_mat100(split, idx)
    else:
        print("Type {} not recognized, will use mat100".format(type))
        x = load_mat100(split, idx)
    if split == "tr":
        y_filename = "Ytr{}.csv".format(idx)
        y_path = os.path.join(os.getcwd(), "data", "original", y_filename)
        y_df = pd.read_csv(y_path)
        y = np.array(y_df.iloc[:, 1].values)
        y = np.where(y == 1, 1, -1)
        return x, y
    return x


def data_wrapper(idx, split, type_list):
    if split == "tr":
        y = None
        xs = []
        for type in type_list:
            x, y = load_data(idx, split, type)
            xs.append(x)
        return xs, y
    else:
        return [load_data(idx, split, type) for type in type_list]


def get_pred_subms(cfg, type):
    predictions = []
    full_log = ""

    for idx in range(3):
        scale = cfg.DATA.scale
        X_tr_l, y_tr = data_wrapper(idx, "tr", type)
        X_te_l = data_wrapper(idx, "te", type)
        print("Preprocessing done")

        # get hyperparams from CV
        model_class = models_dic[cfg.MODEL_NAME]
        best_val_score, best_param = grid_search_cv(model_class, cfg.grid_hparams, X_tr_l, y_tr, cfg.DATA.k_list,
                                                    cfg.N_FOLDS, scale)
        print("Grid search done")

        # scale data
        X_tr_l, X_te_l = scale_list(X_tr_l, X_te_l, scale)

        model = model_class(cfg.DATA.k_list, **best_param)
        model.fit(X_tr_l, y_tr)
        train_score = get_accuracy(model, X_tr_l, y_tr)
        predictions.append(model.predict(X_te_l))
        log = "Split {}: train_acc: {:.3f}, val_acc: {:.3f}, selected parameters: {} \n".format(idx, train_score,
                                                                                                best_val_score,
                                                                                                best_param)
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

    preds_df, result_log = get_pred_subms(cfg, cfg.DATA.type_list)

    os.mkdir(save_dir)
    shutil.copy(cfg_path, save_dir)
    preds_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    with open(os.path.join(save_dir, "log.txt"), "w") as text_file:
        text_file.write(result_log)
    cfg_name = os.path.split(cfg_path)[-1]
    print("Done with {}.".format(cfg_name))
