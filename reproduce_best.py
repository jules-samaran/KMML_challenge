import os

from submissions import subs_wrapper, ensemble_preds


subs_dir = os.path.join(os.getcwd(), "submissions")
cfg_paths = [os.path.join(os.getcwd(), "cfgs", "best_submission_cfg", "cfg_{}.yaml".format(k)) for k in range(1, 4)]

preds_paths = []
for path in cfg_paths:
    preds_paths.append(subs_wrapper(path, subs_dir))

ensemble_preds("best_submission", preds_paths)
