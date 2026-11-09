"""Define stress regimes (R6) for headroom-gated evaluation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
from uav_wireless.experiments import base_config

REGIMES = {
    "nominal": {},
    "single_pad": {"n_pads": 1},
    "small_battery": {"B_max": 15000.0, "E_reserve": 1000.0},
    "low_comm": {"R_comm": 25.0},
    "clustered": {"start_clustered": False},
    "failure": {"failure_enabled": True, "failure_time": 500},
    "multi_failure": {"failure_enabled": True, "failure_time": 300},
}

def get_config(name, **overrides):
    cfg = base_config()
    if name in REGIMES and REGIMES[name] is not None:
        for k, v in REGIMES[name].items():
            setattr(cfg, k, v)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

if __name__ == "__main__":
    for name in REGIMES:
        cfg = get_config(name)
        print(f"{name}: N={cfg.N} M={cfg.M} pads={cfg.n_pads} B_max={cfg.B_max} R_comm={cfg.R_comm} fail={cfg.failure_enabled}")
