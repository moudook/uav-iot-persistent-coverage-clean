import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from stress_regimes import get_config
from uav_wireless.train import baseline_fixed, baseline_greedy, baseline_reactive, baseline_reactive_local
import numpy as np

scenes = list(range(1000, 1020))
cfg = get_config("nominal")
cfg.episode_len = 500
rules = [("Fixed", baseline_fixed), ("Greedy", baseline_greedy),
         ("Reactive", baseline_reactive), ("Reactive-local", baseline_reactive_local)]
for name, fn in rules:
    cks = [float(fn(cfg, steps=500, seed=s)["CK"]) for s in scenes]
    print(f"{name:14s} CK={np.mean(cks):.4f} (sd {np.std(cks, ddof=1):.4f})")