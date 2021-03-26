import os

from submissions import subs_wrapper


subs_dir = os.path.join(os.getcwd(), "submissions")
cfg_paths = [os.path.join(os.getcwd(), "cfgs", "second_best_cfg.yaml")]

for path in cfg_paths:
    _ = subs_wrapper(path, subs_dir)
