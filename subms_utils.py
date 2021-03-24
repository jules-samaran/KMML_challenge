import os
import shutil
import yaml
from easydict import EasyDict as edict

import numpy as np
import pandas as pd
from datetime import datetime

from model import models_dic, get_accuracy
from preprocessing import scale_list
from data_utils import data_wrapper
from crossval import grid_search_cv


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


def ensemble_preds(csv_paths=[]):
    preds = [pd.read_csv(path, header=0)["Bound"].values.reshape((-1, 1)) for path in csv_paths]
    preds_array = np.concatenate(preds, axis=1)
    mean_array = np.mean(preds_array, axis=1)
    ensbl_pred = np.where(mean_array > 0.5, 1, 0)
    pred_df = pd.DataFrame(np.concatenate((np.arange(ensbl_pred.shape[0]).reshape((-1, 1)),
                                           ensbl_pred.reshape((-1, 1))), axis=1),
                           columns=["Id", "Bound"])
    date = datetime.now().strftime("%d_%H:%M:%S:")
    dir_name = "{}_{}".format(date, "ensemble")
    save_dir = os.path.join("submissions/", dir_name)

    os.mkdir(save_dir)
    with open(os.path.join(save_dir, "list_ensemble.txt"), "w") as text_file:
        text_file.write(str(csv_paths))
    pred_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)


def main():
    csv_paths = ["submissions/24_16:59:26:_KRR/predictions.csv", "submissions/24_14:09:24:_SVM/predictions.csv",
                 "submissions/24_18:34:35:_KRR/predictions.csv"]
    ensemble_preds(csv_paths)


if __name__ == "__main__":
    main()
