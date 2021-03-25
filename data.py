import os

import numpy as np
import pandas as pd

from preprocessing import spectrum_phi, mismatch


def load_spectrum(split, idx, k):
    """
    Load saved spectrum features or compute them.

    :param split: str
        tr or te
    :param idx: int
        0, 1 or 2
    :param k: int
        length of kmers

    :return: array
        Array with spectrum features.
    """
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
    """
    Load mat100 features
    :param split: str
        tr or te
    :param idx: int
        0, 1 or 2
    :return: array
        Array with mat100 features.
    """
    suffix = "_mat100"

    x_filename = "X{}{}{}.csv".format(split, idx, suffix)
    x_path = os.path.join(os.getcwd(), "data", "original", x_filename)
    x_df = pd.read_csv(x_path, header=None, sep=" ")
    x = np.array(x_df.values)
    return x


def load_data(idx, split, type):
    """
    Wrapper function to load data.

    :param split: str
        tr or te
    :param idx: int
        0, 1 or 2
    :param type: str
        Type of features to load

    :return: array
        Load features (and labels when train is loaded)

    """
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
    """
    Big wrapper function to load all data when more than one type of features.

    :param split: str
        tr or te
    :param idx: int
        0, 1 or 2
    :param type_list: list of str
        Types of features to load

    :return: list of arrays
        List with features in different arrays depending on feature type.
    """
    if split == "tr":
        y = None
        xs = []
        for type in type_list:
            x, y = load_data(idx, split, type)
            xs.append(x)
        return xs, y
    else:
        return [load_data(idx, split, type) for type in type_list]
