"""
Energy-flow feasibility certificate for persistent K-coverage.

Extends the spatial packing certificate (feasibility.py) with an energy and
charging-throughput bound.  The spatial certificate asks "how many UAVs are
needed to K-cover the region"; this one asks "given the battery and pad
budget, how many UAV-seconds can the fleet actually spend airborne, and
therefore what is the maximum time-averaged K-coverage it can sustain".

Three quantities are reported:

  1. Steady-state bound (T -> infinity): the fleet can keep airborne at most
     S * P_pad_net / P_air UAVs, so persistent K-coverage needs
     N_min <= S * P_pad_net / P_air.  If this fails, no policy, however
     clever, can sustain the mission; the configuration is not persistent.

  2. Finite-horizon bound (T = episode length): even over a finite mission
     the airborne UAV-seconds are capped by (stored energy + pad energy) / P_air.

  3. Energy-aware ceiling: a small linear-program (fluid relaxation) that
     maximises the time-averaged airborne fleet (and hence an upper bound on
     CK) subject to the per-UAV energy budget, the pad throughput, and the
     no-fly-while-charging constraint.  It is the energy analogue of the
     teleporting placement oracle.

Usage:
    python scripts/energy_certificate.py
    python scripts/energy_certificate.py --T 2000 --n-pads 4 --B-max 60000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_impl"))
from uav_wireless.env import SimConfig


def params(cfg: SimConfig) -> dict:
    p_air = cfg.P_hover + cfg.P_elec + cfg.P_tx
    p_pad_gross = cfg.charge_rate * cfg.B_max / cfg.dt
    p_pad_net = p_pad_gross - (cfg.P_elec + cfg.P_tx)
    n_min = 4 * cfg.K
    return dict(
        N=cfg.N, K=cfg.K, M=cfg.M, side=cfg.side,
        B_max=cfg.B_max, E_reserve=cfg.E_reserve,
        charge_rate=cfg.charge_rate, n_pads=cfg.n_pads,
        p_air=p_air, p_pad_gross=p_pad_gross, p_pad_net=p_pad_net,
        n_min=n_min, dt=cfg.dt, episode_len=cfg.episode_len,
    )


def steady_state(p: dict) -> dict:
    n_air_ss = p["n_pads"] * p["p_pad_net"] / p["p_air"]
    ceiling_ss = min(1.0, n_air_ss / p["n_min"])
    feasible = n_air_ss >= p["n_min"]
    return dict(n_air_ss=n_air_ss, ceiling_ss=ceiling_ss, feasible=feasible,
                deficit=p["n_min"] - n_air_ss)


def finite_horizon(p: dict, T: float, e0_frac: float = 1.0) -> dict:
    e_stored = p["N"] * e0_frac * p["B_max"]
    e_pad = p["n_pads"] * p["p_pad_net"] * T
    e_usable = e_stored + e_pad - p["N"] * p["E_reserve"]
    air_seconds = max(0.0, e_usable / p["p_air"])
    ceiling = min(1.0, air_seconds / (p["n_min"] * T))
    return dict(T=T, e_stored=e_stored, e_pad=e_pad, e_usable=e_usable,
                air_seconds=air_seconds, ceiling=ceiling,
                avg_airborne=air_seconds / T)


def energy_lp(p: dict, T: float, e0_frac: float = 1.0) -> dict:
    N = p["N"]
    E0 = e0_frac * p["B_max"]
    reserve = p["E_reserve"]
    P_air = p["p_air"]
    P_pad = p["p_pad_net"]
    S = p["n_pads"]

    nvar = 2 * N
    c_obj = np.zeros(nvar)
    c_obj[:N] = -1.0

    A_ub = []
    b_ub = []
    for i in range(N):
        row = np.zeros(nvar)
        row[i] = 1.0
        row[N + i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
    row = np.zeros(nvar)
    row[N:] = 1.0
    A_ub.append(row)
    b_ub.append(float(S))
    for i in range(N):
        row = np.zeros(nvar)
        row[i] = P_air * T
        row[N + i] = -P_pad * T
        A_ub.append(row)
        b_ub.append(E0 - reserve)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, 1.0)] * nvar

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return dict(success=False, message=res.message)
    a = res.x[:N]
    c = res.x[N:]
    air = float(a.sum())
    ceiling = min(1.0, air / p["n_min"])
    return dict(success=True, air=air, ceiling=ceiling, a=[round(x, 3) for x in a],
                c=[round(x, 3) for x in c], n_min=p["n_min"], status=res.status)


def report(cfg: SimConfig, T: float, e0_frac: float = 1.0) -> dict:
    p = params(cfg)
    ss = steady_state(p)
    fh = finite_horizon(p, T, e0_frac)
    lp = energy_lp(p, T, e0_frac)
    return dict(params=p, steady_state=ss, finite_horizon=fh, lp=lp)


def print_report(r: dict, label: str = "") -> None:
    p = r["params"]
    ss = r["steady_state"]
    fh = r["finite_horizon"]
    lp = r["lp"]
    print("=" * 78)
    if label:
        print(label)
    print(f"N={p['N']} K={p['K']} M={p['M']} side={p['side']} n_pads={p['n_pads']} "
          f"B_max={p['B_max']:.0f} charge_rate={p['charge_rate']} E_reserve={p['E_reserve']:.0f}")
    print(f"P_air (hover+elec+tx) = {p['p_air']:.2f} W   "
          f"P_pad gross = {p['p_pad_gross']:.2f} W   net = {p['p_pad_net']:.2f} W")
    print(f"Spatial certificate: N_min = 4K = {p['n_min']} UAVs needed for full K-coverage")
    print()
    print("Steady-state (T -> inf):")
    print(f"  max sustainable airborne UAVs = S*P_pad_net/P_air = "
          f"{p['n_pads']}*{p['p_pad_net']:.2f}/{p['p_air']:.2f} = {ss['n_air_ss']:.3f}")
    print(f"  persistent CK ceiling = {ss['ceiling_ss']:.3f}   "
          f"feasible = {ss['feasible']}   deficit = {ss['deficit']:.2f} UAVs")
    print()
    print(f"Finite horizon T = {fh['T']:.0f} s (e0_frac = {fh['e_stored']/p['N']/p['B_max']:.2f} of full charge):")
    print(f"  stored = {fh['e_stored']/1000:.1f} kJ   pad = {fh['e_pad']/1000:.1f} kJ   "
          f"usable = {fh['e_usable']/1000:.1f} kJ")
    print(f"  airborne UAV-seconds = {fh['air_seconds']:.0f}   avg airborne = {fh['avg_airborne']:.2f}   "
          f"ceiling = {fh['ceiling']:.3f}")
    print()
    print("Energy-aware LP ceiling (fluid relaxation):")
    if lp.get("success"):
        print(f"  max avg airborne UAVs = {lp['air']:.3f}   ceiling CK = {lp['ceiling']:.3f}")
        print(f"  a (airborne frac per UAV) = {lp['a']}")
        print(f"  c (charging frac per UAV) = {lp['c']}")
    else:
        print("  LP failed:", lp.get("message"))
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=None, help="mission length in seconds (default: episode_len*dt)")
    ap.add_argument("--n-pads", type=int, default=None)
    ap.add_argument("--B-max", type=float, default=None)
    ap.add_argument("--charge-rate", type=float, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--e0-frac", type=float, default=1.0, help="initial charge fraction (1.0 = full)")
    ap.add_argument("--sweep", action="store_true", help="sweep a few alternative configs")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    def make_cfg(overrides: dict) -> SimConfig:
        base = SimConfig(N=8, M=20, K=2, side=80.0, R_cov=35.0, R_comm=40.0,
                         episode_len=2000, ego_sensors=True, n_ego=3)
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    overrides = {}
    if a.n_pads is not None: overrides["n_pads"] = a.n_pads
    if a.B_max is not None: overrides["B_max"] = a.B_max
    if a.charge_rate is not None: overrides["charge_rate"] = a.charge_rate
    if a.N is not None: overrides["N"] = a.N

    cfg = make_cfg(overrides)
    T = a.T if a.T is not None else cfg.episode_len * cfg.dt

    r = report(cfg, T, a.e0_frac)
    print_report(r, label="CONFIG UNDER TEST")

    if a.sweep:
        print("\n" + "#" * 78)
        print("SWEEP: alternative configurations")
        print("#" * 78)
        variants = [
            ("baseline (N=8, S=2, B=30k, cr=0.0002)", {}),
            ("more pads (S=4)", {"n_pads": 4}),
            ("bigger packs (B=60k)", {"B_max": 60000.0}),
            ("faster charge (cr=0.0004)", {"charge_rate": 0.0004}),
            ("more pads + faster charge", {"n_pads": 4, "charge_rate": 0.0004}),
            ("more UAVs (N=12)", {"N": 12}),
            ("more pads + bigger packs", {"n_pads": 4, "B_max": 60000.0}),
            ("all three (N=12, S=4, B=60k, cr=0.0004)", {"N": 12, "n_pads": 4, "B_max": 60000.0, "charge_rate": 0.0004}),
        ]
        rows = []
        for label, ov in variants:
            c = make_cfg(ov)
            rr = report(c, T, a.e0_frac)
            ss = rr["steady_state"]
            lp = rr["lp"]
            rows.append((label, c.N, c.n_pads, c.B_max, c.charge_rate,
                         ss["n_air_ss"], ss["ceiling_ss"], ss["feasible"],
                         lp.get("ceiling", float("nan"))))
        print(f"{'variant':42s} {'N':>3s} {'S':>2s} {'Bmax':>6s} {'cr':>7s} {'ss_air':>7s} {'ss_ceil':>7s} {'feas':>5s} {'lp_ceil':>7s}")
        for row in rows:
            print(f"{row[0]:42s} {row[1]:3d} {row[2]:2d} {row[3]:6.0f} {row[4]:7.4f} "
                  f"{row[5]:7.3f} {row[6]:7.3f} {str(row[7]):>5s} {row[8]:7.3f}")

    out = a.out or os.path.join(os.path.dirname(__file__), "..", "ml_impl", "runs", "energy_certificate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"T": T, "e0_frac": a.e0_frac, "report": r}, f, indent=1, default=str)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
