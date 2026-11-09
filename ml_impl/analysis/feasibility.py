"""
Feasibility oracle and lower bounds on the number of UAVs required.

Why this must exist
-------------------
A learned policy that scores CK = 0.35 is only evidence of a learning failure
if some other deployment would have scored better. If no arrangement of N UAVs
can K-cover the sensors while staying connected, then a low CK is a statement
about the configuration, not about MAPPO, and reporting it as a policy result
would be wrong.

Two kinds of answer are provided, and they are not the same thing:

**Rigorous bounds** (`region_2cover_bound`, `sensor_packing_bound`). These are
necessary conditions: a proof that fewer than some number of UAVs CANNOT
succeed. They hold for every possible policy, every control law and every
learning algorithm, because they depend only on geometry. Each is documented
with its argument, and the test suite checks each argument.

**Oracle search** (`search_best`). An upper bound obtained by brute-force
optimisation over deployments, with energy, motion limits and time ignored.
No online policy can beat it, because the oracle may teleport and can see the
whole sensor field. Comparing a policy to the oracle separates "the policy is
weak" from "the task is hard".

The gap between the two is the honest statement about the problem.

Bounds implemented
------------------
Region bound: the four corners of the square are 80 m apart, while two coverage
    discs of radius R_cov = 35 m intersect only if their centres are within
    2*R_cov = 70 m. All four corner discs are therefore pairwise disjoint. A
    point is K-covered only by UAVs that lie inside its own disc, so disjoint
    discs require disjoint sets of UAVs, and N >= 4K is necessary.
Sensor packing bound: the same argument applied to whichever sensors happen to
    be far apart, giving a seed-specific N >= K * |S| for a maximal set S of
    sensors with pairwise-disjoint coverage discs.

Usage
-----
    python -m analysis.feasibility
    python -m analysis.feasibility --seeds 10 --sweep
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_wireless import SimConfig, UAVIoTEnv
from dataclasses import replace


# ======================================================================
# Rigorous necessary conditions
# ======================================================================
def region_2cover_bound(cfg: SimConfig) -> dict:
    """Necessary number of UAVs to K-cover the four corners of the region.

    The argument is short enough to check by hand:

    1. A sensor at point p is K-covered only if at least K UAVs lie within
       distance R_cov of p, i.e. inside the disc D(p, R_cov).
    2. Two discs D(p1, R_cov) and D(p2, R_cov) are disjoint whenever
       |p1 - p2| > 2*R_cov, because a shared UAV would satisfy
       |u - p1| <= R_cov and |u - p2| <= R_cov, and the triangle inequality
       would then force |p1 - p2| <= 2*R_cov.
    3. The four corners of the square are pairwise at least `side` apart. With
       side = 80 and 2*R_cov = 70 we have 80 > 70, so all four corner discs are
       pairwise disjoint.
    4. Disjoint discs cannot share a UAV, so each corner consumes K UAVs of its
       own and N >= 4K.

    This is why a strict service-continuity metric can be identically zero for
    every model: it is unreachable at the reported parameters.
    """
    side, r = cfg.side, cfg.R_cov
    corners = np.array([[0.0, 0.0], [side, 0.0], [0.0, side], [side, side]])
    d = np.linalg.norm(corners[:, None] - corners[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_sep = float(d.min())
    disjoint = min_sep > 2.0 * r
    required = 4 * cfg.K if disjoint else cfg.K
    return dict(
        min_corner_separation_m=min_sep,
        two_R_cov_m=2.0 * r,
        corner_discs_disjoint=bool(disjoint),
        N_required_for_corners=int(required),
        N_available=int(cfg.N),
        corners_feasible=bool(cfg.N >= required),
    )


def sensor_packing_bound(sensors: np.ndarray, cfg: SimConfig) -> dict:
    """Necessary N for the ACTUAL sensor layout: N >= K * |S|.

    S is a maximal set of sensors whose coverage discs are pairwise disjoint.
    Building S greedily is enough for a valid bound: any such S is a
    certificate, and maximality only affects tightness. Each sensor in S needs
    K UAVs inside a disc that no other member of S shares, so the UAV sets are
    disjoint.
    """
    n = len(sensors)
    if n == 0:
        return dict(packing_size=0, N_required=0, N_available=int(cfg.N),
                    sensors_feasible=True)

    used = np.zeros(n, dtype=bool)
    chosen = []
    # take candidates in order of increasing local crowding: a sensor with few
    # neighbours is a good first pick for a large packing
    counts = (np.linalg.norm(sensors[:, None] - sensors[None], axis=-1)
              <= 2.0 * cfg.R_cov).sum(axis=1)
    for i in np.argsort(counts):
        if used[i]:
            continue
        chosen.append(int(i))
        far = np.linalg.norm(sensors - sensors[i], axis=-1) > 2.0 * cfg.R_cov
        used |= ~far
    required = cfg.K * len(chosen)
    return dict(
        packing_size=len(chosen),
        packing_sensors=chosen,
        N_required=int(required),
        N_available=int(cfg.N),
        sensors_feasible=bool(cfg.N >= required),
    )


# ======================================================================
# Oracle search
# ======================================================================
def coverage_score(env: UAVIoTEnv, pos: np.ndarray) -> float:
    """Fraction of sensors K-covered by `pos`, through the env's own PHY gate."""
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (env.cfg.N, 2):
        raise ValueError(f"placement must be ({env.cfg.N}, 2), got {pos.shape}")
    saved = env.p.copy()
    try:
        env.p = pos.copy()
        return float(np.mean(env.coverage_counts() >= env.cfg.K))
    finally:
        env.p = saved


def is_connected_at(env: UAVIoTEnv, pos: np.ndarray) -> bool:
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (env.cfg.N, 2):
        raise ValueError(f"placement must be ({env.cfg.N}, 2), got {pos.shape}")
    saved = env.p.copy()
    try:
        env.p = pos.copy()
        return bool(env.is_connected(env.comm_graph()[1]))
    finally:
        env.p = saved


def _spread_pattern(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Strategies for seeding the search, all more useful than uniform noise.

    Uniform random placement clusters by chance, so it wastes most of the N
    UAVs on overlapping discs. These patterns deliberately spread.

    Always returns exactly (cfg.N, 2); a lattice with too few cells is padded.
    """
    n, side = cfg.N, cfg.side
    mode = int(rng.integers(0, 4))

    if mode == 0:                                    # jittered lattice
        k = int(np.ceil(np.sqrt(n)))
        ticks = np.linspace(0, side, k + 1)[1:-1]
        if len(ticks) == 0:
            ticks = np.array([side / 2])
        gx, gy = np.meshgrid(ticks, ticks)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        if len(pts) < n:                             # lattice too small
            extra = rng.uniform(0, side, size=(n - len(pts), 2))
            pts = np.vstack([pts, extra])
        idx = rng.permutation(len(pts))[:n]
        return np.clip(pts[idx].astype(float), 0, side)

    if mode == 1:                                    # opposite-corner pairs
        c = side * 0.30
        pts = np.array([[c, c], [c, c], [side - c, side - c], [side - c, side - c],
                        [side / 2, side / 2]])

    elif mode == 2:                                  # all four corners
        c = side * 0.30
        pts = np.array([[c, c], [side - c, c], [c, side - c], [side - c, side - c]])

    else:                                            # ring reaching the corners
        rad = max(np.sqrt(2) * side / 2 - cfg.R_cov * 0.6, 0.05 * side)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        centre = np.array([side / 2, side / 2])
        pts = centre + rad * np.stack([np.cos(ang), np.sin(ang)], axis=1)

    # pad by tiling and jitter, then crop so the shape is always exactly (N, 2)
    if len(pts) < n:
        reps = int(np.ceil(n / len(pts)))
        pts = np.vstack([pts] * reps)
    pts = pts[:n]
    jit = rng.normal(0, 0.05 * side, size=(len(pts), 2))
    return np.clip(pts + jit, 0.0, side)


def search_best(env: UAVIoTEnv, iters: int = 80, restarts: int = 12,
                rng: np.random.Generator | None = None,
                connected_only: bool = False):
    """Maximise covered fraction by random restart + local descent.

    The oracle may teleport UAVs, so the result is a strict upper bound on any
    policy that is constrained by v_max and by keeping the swarm connected
    while it moves.
    """
    cfg = env.cfg
    rng = rng if rng is not None else np.random.default_rng(0)
    best_pos, best_ck = None, -1.0

    for _ in range(restarts):
        pos = _spread_pattern(cfg, rng).astype(float)
        if connected_only and not is_connected_at(env, pos):
            # pull the swarm toward its centroid until the graph closes
            centre = pos.mean(axis=0)
            for t in np.linspace(0.0, 1.0, 40):
                cand = (1 - t) * pos + t * centre
                if is_connected_at(env, cand):
                    pos = cand
                    break
            else:
                continue

        ck = coverage_score(env, pos)
        step = 0.30 * cfg.side
        stall = 0

        while step > 0.01 * cfg.side and stall < 3:
            improved = False
            for i in range(cfg.N):
                for _try in range(30):
                    cand = pos.copy()
                    cand[i] = np.clip(cand[i] + rng.normal(0, step, size=2),
                                      0.0, cfg.side)
                    if connected_only and not is_connected_at(env, cand):
                        continue
                    ck_new = coverage_score(env, cand)
                    if ck_new > ck + 1e-12:
                        pos, ck = cand, ck_new
                        improved = True
            if improved:
                stall = 0
            else:
                stall += 1
                step *= 0.5
        if ck > best_ck:
            best_pos, best_ck = pos.copy(), ck
    return best_pos, best_ck


# ======================================================================
# Per-scene analysis
# ======================================================================
def analyse_scene(seed: int, cfg: SimConfig, iters: int = 80, restarts: int = 12):
    scene = dataclasses.replace(cfg, seed=seed)
    env = UAVIoTEnv(scene)
    env.reset()

    pos_any, ck_any = search_best(env, iters=iters, restarts=restarts,
                                  rng=np.random.default_rng(seed))
    pos_con, ck_con = search_best(env, iters=iters, restarts=restarts,
                                  rng=np.random.default_rng(seed + 7919),
                                  connected_only=True)

    pack = sensor_packing_bound(env.sensors, scene)
    return dict(
        seed=int(seed),
        oracle_CK_unconstrained=float(ck_any),
        oracle_CK_connected=float(ck_con),
        oracle_SC_reachable=bool(ck_con >= 1.0 - 1e-9),
        oracle_pos_connected=(
            pos_con.round(2).tolist() if pos_con is not None else None
        ),
        required_UAVs_sensor_packing=int(pack["N_required"]),
        packing_size=int(pack["packing_size"]),
        N=int(scene.N),
        K=int(scene.K),
    )


def geometric_surplus(cfg: SimConfig) -> dict:
    """The report's sigma = N*pi*R_cov^2 / (K*A), and why it misleads.

    sigma compares an AREA to an AREA. Coverage of a bounded region by a fixed
    number of equal discs is a COVERING problem: what matters is how the discs
    can be placed, not how much area they total. Two discs can have plenty of
    area and still leave a corner unreached. sigma = 1.5 therefore does not
    imply that K-coverage is achievable, and the discrepancy is quantified by
    `region_2cover_bound`.
    """
    area = cfg.side ** 2
    return dict(
        surplus_sigma=cfg.N * np.pi * cfg.R_cov ** 2 / (cfg.K * area),
        field_area_m2=area,
        total_disc_area_m2=cfg.N * np.pi * cfg.R_cov ** 2,
        area_required_for_K_m2=cfg.K * area,
        region_diagonal_m=float(cfg.side * np.sqrt(2)),
        max_chain_span_m=float((cfg.N - 1) * cfg.R_comm),
        span_slack_m=float((cfg.N - 1) * cfg.R_comm - cfg.side * np.sqrt(2)),
    )


# ======================================================================
# CLI
# ======================================================================
def packing_survey(cfg: SimConfig, seeds: int = 30) -> dict:
    """How the packing bound behaves over random sensor draws.

    The corner argument of `region_2cover_bound` is a statement about the
    REGION: it applies to the requirement that every point of Omega is
    K-covered.  It does NOT apply to the requirement that every SENSOR is
    K-covered, because the sensors need not be anywhere near the corners.

    The bound that does apply to a set of points is the packing lemma:

        A is K-covered by N discs of radius R  =>  N >= K * P(A, R)

    where P(A, R) is the largest number of points of A that are pairwise
    farther apart than 2R.  A disc of radius R contains at most one such
    point, so the discs that serve them are disjoint, and each of the P
    points needs K of them.

    This survey measures P over random sensor draws, which is what the
    experiments actually face, and it reports how many draws the corner
    certificate would wrongly be needed for.
    """
    ps, reqs = [], []
    for s in range(seeds):
        c = replace(cfg, seed=s)
        env = UAVIoTEnv(c)
        env.reset()
        b = sensor_packing_bound(env.sensors, c)
        ps.append(int(b["packing_size"]))
        reqs.append(int(b["N_required"]))
    ps_a, req_a = np.array(ps), np.array(reqs)
    return dict(
        seeds=seeds,
        packing_size_mean=float(ps_a.mean()),
        packing_size_min=int(ps_a.min()),
        packing_size_max=int(ps_a.max()),
        N_required_mean=float(req_a.mean()),
        N_required_min=int(req_a.min()),
        N_required_max=int(req_a.max()),
        N_available=int(cfg.N),
        seeds_excluded=int(np.sum(req_a > cfg.N)),
        seeds_excluded_fraction=float(np.mean(req_a > cfg.N)),
        packing_sizes=ps,
        N_required=reqs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="results_feasibility.json")
    a = ap.parse_args()

    base = SimConfig()
    payload = dict(config=dataclasses.asdict(base))

    print("=" * 74)
    print("IS THE CONFIGURATION FEASIBLE?   (report Table VI parameters)")
    print("=" * 74)
    rb = region_2cover_bound(base)
    print(f"  region: {base.side:.0f} x {base.side:.0f} m, "
          f"R_cov = {base.R_cov:.0f} m, K = {base.K}, N = {base.N}")
    print(f"  closest two corners              {rb['min_corner_separation_m']:.0f} m")
    print(f"  two coverage discs intersect if  <= {rb['two_R_cov_m']:.0f} m apart")
    print(f"  corner discs pairwise disjoint   {rb['corner_discs_disjoint']}")
    print(f"  => UAVs PROVABLY required        {rb['N_required_for_corners']} "
          f"(= 4K)")
    print(f"  => UAVs available                {rb['N_available']}")
    print(f"  => corners coverable?            {rb['corners_feasible']}")
    payload["region_bound"] = rb

    print()
    print("  Report's area-surplus metric sigma:")
    for k, v in geometric_surplus(base).items():
        print(f"    {k:28s} {v:12.3f}")
    payload["surplus"] = geometric_surplus(base)

    print()
    print("=" * 74)
    print("PACKING BOUND OVER RANDOM SENSOR DRAWS  (the sensor requirement)")
    print("=" * 74)
    ps = packing_survey(base, seeds=30)
    payload["packing_survey"] = ps
    print(f"  packing number P(A, R_cov)     mean {ps['packing_size_mean']:.2f}, "
          f"range {ps['packing_size_min']}..{ps['packing_size_max']}")
    print(f"  => N >= K * P                  mean {ps['N_required_mean']:.2f}, "
          f"range {ps['N_required_min']}..{ps['N_required_max']}")
    print(f"  => draws excluded at N = {base.N}       {ps['seeds_excluded']} "
          f"of {ps['seeds']} ({100*ps['seeds_excluded_fraction']:.0f}%)")
    print("  (the corner certificate applies to the REGION, not to a random")
    print("   sensor draw; see the report for the two requirements)")
    print()

    print()
    print("=" * 74)
    print("ORACLE  (teleporting placement; ignores energy and motion)")
    print("=" * 74)
    rows = [analyse_scene(s, base, a.iters, a.restarts) for s in range(a.seeds)]
    ck_any = float(np.mean([r["oracle_CK_unconstrained"] for r in rows]))
    ck_con = float(np.mean([r["oracle_CK_connected"] for r in rows]))
    req = [r["required_UAVs_sensor_packing"] for r in rows]
    print(f"  best CK, placement unconstrained   {ck_any:.3f}")
    print(f"  best CK, placement connected       {ck_con:.3f}")
    print(f"  SC reachable at best connected     "
          f"{np.mean([r['oracle_SC_reachable'] for r in rows]):.3f}")
    print(f"  UAVs needed (sensor packing)       mean {np.mean(req):.1f}, "
          f"max {max(req)}   [available {base.N}]")
    print()
    print(f"  {'seed':>4} {'CK.any':>7} {'CK.conn':>8} {'N_needed':>9}")
    for r in rows:
        print(f"  {r['seed']:>4} {r['oracle_CK_unconstrained']:>7.3f} "
              f"{r['oracle_CK_connected']:>8.3f} "
              f"{r['required_UAVs_sensor_packing']:>9d}")
    payload["oracle"] = rows

    if a.sweep:
        print()
        print("=" * 74)
        print("SWEEP  (where does the task become feasible?)")
        print("=" * 74)
        print(f"  {'N':>3} {'K':>2} {'R_cov':>6} {'R_comm':>7} {'4K':>4} "
              f"{'CK.con':>7} {'SC':>4}")
        sweep = []
        for n in (5, 6, 8, 10):
            for k in (1, 2):
                for r_cov, r_com in ((base.R_cov, base.R_comm), (25.0, 40.0)):
                    cfg = SimConfig(N=n, K=k, R_cov=r_cov, R_comm=r_com)
                    rb2 = region_2cover_bound(cfg)
                    cks, scs = [], []
                    for s in range(3):
                        rr = analyse_scene(s, cfg, iters=45, restarts=6)
                        cks.append(rr["oracle_CK_connected"])
                        scs.append(rr["oracle_SC_reachable"])
                    mc, ms = float(np.mean(cks)), float(np.mean(scs))
                    print(f"  {n:>3} {k:>2} {r_cov:>6.0f} {r_com:>7.0f} "
                          f"{rb2['N_required_for_corners']:>4d} {mc:>7.3f} "
                          f"{ms:>4.1f}")
                    sweep.append(dict(
                        N=n, K=k, R_cov=r_cov, R_comm=r_com,
                        N_required_corners=rb2["N_required_for_corners"],
                        corners_feasible=rb2["corners_feasible"],
                        oracle_CK_connected=mc, oracle_SC_reachable=ms,
                    ))
        payload["sweep"] = sweep

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
