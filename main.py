import os

from utils import subs_wrapper

subs_dir = os.path.join(os.getcwd(), "submissions")
cfg_paths = [os.path.join(os.getcwd(), "cfgs", "cfg_krr.yaml")]

for path in cfg_paths:
    subs_wrapper(path, subs_dir)
