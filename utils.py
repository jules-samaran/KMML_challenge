import os
import yaml
from easydict import EasyDict as edict

import numpy as np
import pandas as pd
from datetime import datetime


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
