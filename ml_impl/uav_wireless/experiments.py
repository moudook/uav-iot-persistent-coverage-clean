"""
Experiment design matrix (Table X / Section VII).

Runs the full comparison: Fixed, Greedy, Reactive, MAPPO (proposed),
Vanilla MAPPO (ablation) across 8 scenarios x 5 seeds, and prints a
metrics table. Results are saved to CSV.
"""

from __future__ import annotations

import os
import numpy as np

from .env import SimConfig, UAVIoTEnv
from .train import (
    train_mappo, collect_rollout,
    baseline_fixed, baseline_greedy, baseline_reactive,
    baseline_reactive_local,
)


def base_config(**overrides) -> SimConfig:
    """The one canonical experimental configuration.

    Every script must build its configuration from here, so that the
    numbers in the paper, the numbers in the CSVs and the numbers in the
    audit all describe the same experiment.

    Choices, and why:

    N = 8, K = 2, R_cov = 35 m
        The corner-packing bound requires N >= 4K, so N = 8 is the smallest
        feasible fleet for K = 2 on this field.

    start_clustered = True
        A centred ring deployment already covers 17 of 20 sensors at t = 0,
        so every controller -- including one that never moves -- scores
        CK = 0.850 and the comparison cannot separate them.  The clustered
        deployment is connected at t = 0 and covers 3 of 20, so the
        controller has real work to do.

    episode_len = 2000 steps (400 s of flight)
        At 400 steps the pack never depletes, so the return-to-charge
        transition never fires, the charging arbitration is never
        exercised and the reserve UAV has nothing to recover.  At 2000
        steps the constraint binds (283 return transitions and 0.7 stranded
        UAVs per episode under the Reactive heuristic) while the mission
        remains achievable.

    B_max = 30 000 J, charge_rate = 0.0002/step
        The pack size and charge rate of the report's parameter table; the
        episode length is what changed.

    use_agent_id, map_grid = 4, use_pad_bearing
        The observation must identify the agent for a shared policy to
        specialise, must carry a summary of where the coverage deficit is
        (the reactive heuristic is given the EXACT deficit set, so the
        comparison is only like-for-like if the learned policy has a
        broadcast summary of it too), and must give the direction of the
        charging pad.  The map summary is MEASURED to hurt the learned
        controller at this budget -- the policy follows the deficit, because
        a location without an assignment is not actionable -- and the
        measurement is reported in the results, together with the variant
        that omits it.
    """
    cfg = SimConfig(
        N=8, K=2, R_cov=35.0, M=20, episode_len=2000, seed=0,
        B_max=30_000.0, charge_rate=0.0002,
        lam_B=2.0,
        start_clustered=True,
        use_agent_id=True, map_grid=4, use_pad_bearing=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def scenario_cfgs(base: SimConfig):
    """S1..S8 -> list of (id, research question, [configs])."""
    import copy

    def cp(**kw):
        c = copy.deepcopy(base)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    scen = []
    scen.append(("S1", "sensor density",
                 [cp(M=10, seed=s) for s in range(2)] +
                 [cp(M=40, seed=s) for s in range(2)]))
    scen.append(("S2", "fleet size N",
                 [cp(N=3, seed=s) for s in range(2)] +
                 [cp(N=8, seed=s) for s in range(2)]))
    scen.append(("S3", "battery capacity",
                 [cp(B_max=15_000.0, charge_rate=0.0004, seed=s) for s in range(2)] +
                 [cp(B_max=60_000.0, seed=s) for s in range(2)]))
    scen.append(("S4", "charging pads S",
                 [cp(n_pads=1, seed=s) for s in range(2)] +
                 [cp(n_pads=4, seed=s) for s in range(2)]))
    scen.append(("S5", "R_comm",
                 [cp(R_comm=25.0, seed=s) for s in range(2)] +
                 [cp(R_comm=55.0, seed=s) for s in range(2)]))
    scen.append(("S6", "R_cov",
                 [cp(R_cov=28.0, seed=s) for s in range(2)] +
                 [cp(R_cov=45.0, seed=s) for s in range(2)]))
    scen.append(("S7", "failure time",
                 [cp(failure_enabled=True, failure_time=200, seed=s) for s in range(2)] +
                 [cp(failure_enabled=True, failure_time=700, seed=s) for s in range(2)]))
    scen.append(("S8", "combined stress",
                 [cp(failure_enabled=True, failure_time=500, n_pads=1,
                     B_max=15_000.0, charge_rate=0.0004, R_comm=30.0, R_cov=30.0, seed=s)
                  for s in range(2)]))
    scen.append(("S9", "initial deployment",
                 [cp(start_clustered=True, seed=s) for s in range(2)] +
                 [cp(start_clustered=False, seed=s) for s in range(2)]))
    return scen


def eval_agent(agent, cfg, steps=1000, seed=0, deterministic=False):
    """Roll out the learned policy and return the episode summary.

    The default is the SAMPLED policy, because Eq. (13) of the report
    specifies a_i(t) ~ pi_theta(a_i | o_i(t)): the deployed policy draws
    from the categorical distribution, it does not take the argmax.  The
    difference between the two is not cosmetic and is reported: with a
    shared actor the deterministic policy tends to select the SAME action
    for every UAV, so the swarm translates as a rigid body and never
    disperses, while the sampled policy breaks that tie by chance and
    deploys.  `deterministic=True` selects the argmax policy for
    comparison.
    """
    import copy
    c = copy.deepcopy(cfg)
    c.seed = seed
    # The policy samples, and the sampling draws from the global torch RNG.
    # Seed it from the evaluation seed alone, so that the evaluation is a
    # function of (policy, scene) and every controller sees the same stream
    # on the same scene.  See uav_wireless/seeding.py.
    from .seeding import seed_evaluation
    seed_evaluation(seed)
    env = UAVIoTEnv(c)
    obs, st = env.reset()
    for _ in range(steps):
        import numpy as np
        mask = np.stack(env.action_masks())
        for i in range(c.N):
            if mask[i].sum() == 0:
                mask[i, 0] = True
                mask[i] = mask[i] | (mask[i] == 0)
        acts = agent.act(obs, mask, deterministic=deterministic)
        obs, st, r, done, _ = env.step(acts)
        if done:
            break
    return env.summary()


def run_base_comparison(base: SimConfig, out_csv="results_main.csv",
                        eval_seeds=5, train=True, agent=None, learned=None):
    """Train MAPPO on the base config, compare against the rule baselines.

    The baselines are evaluated on the SAME seeds as the learned policy.
    (They previously accepted a seed and ignored it, so a "five-seed"
    comparison ran one scene five times for the baselines while the learned
    policy was evaluated on five different scenes -- which flatters the
    learned policy whenever the extra seeds happen to be easier.)

    `learned` carries the learned controller's numbers when they have
    already been measured over several TRAINING seeds, which is the
    protocol the paper describes: a controller is trained N times and
    reported as the mean over the N runs, with the seed-to-seed spread.
    Passing a single `agent` instead reports that one policy, which is
    what this table did before and is NOT the protocol the text describes.
    """
    import csv
    rows = []

    if agent is None and train:
        print("\n=== Training proposed MAPPO on base config ===")
        agent, hist = train_mappo(base, episodes=120, n_steps=400, seed=0,
                                  log_every=20)
        agent.save("mappo_base.pt")

    print(f"\n=== Rule-based baselines ({eval_seeds} seeds) ===")
    for name, fn in [("Fixed", baseline_fixed),
                     ("Greedy", baseline_greedy),
                     ("Reactive", baseline_reactive),
                     # the same rule WITHOUT the privileged global coverage
                     # state: the information-asymmetry arm
                     ("Reactive-local", baseline_reactive_local)]:
        sums = [fn(base, steps=base.episode_len, seed=s)
                for s in range(eval_seeds)]
        agg = {k: float(np.mean([s[k] for s in sums])) for k in sums[0]}
        std = {k: float(np.std([s[k] for s in sums])) for k in sums[0]}
        print(f"{name:10s} CK={agg['CK']:.3f}±{std['CK']:.3f} "
              f"AG={agg['AG']:.3f} SC={agg['SC']:.3f} "
              f"CI={agg['CovIntegral']:.3f} E={agg['Energy']:.0f}")
        rows.append(dict(model=name, **agg))

    if learned:
        # the learned controller, measured over several training seeds
        for name in ("MAPPO", "MAPPO-argmax"):
            r = learned.get(name)
            if r is None:
                continue
            print(f"{name:10s} CK={r['CK']:.3f}+-{r.get('CK_std', 0.0):.3f} "
                  f"AG={r['AG']:.3f} SC={r['SC']:.3f} "
                  f"CI={r['CovIntegral']:.3f} E={r.get('Energy', 0.0):.0f} "
                  f"({r.get('train_seeds', '?')} training seeds)")
            rows.append(dict(model=name, **r))
    else:
        print(f"\n=== MAPPO (proposed) evaluation ({eval_seeds} seeds) ===")
        evs = [eval_agent(agent, base, steps=base.episode_len, seed=s)
               for s in range(eval_seeds)]
        agg = {k: float(np.mean([s[k] for s in evs])) for k in evs[0]}
        std = {k: float(np.std([s[k] for s in evs])) for k in evs[0]}
        print(f"{'MAPPO':10s} CK={agg['CK']:.3f}+-{std['CK']:.3f} "
              f"AG={agg['AG']:.3f} SC={agg['SC']:.3f} "
              f"CI={agg['CovIntegral']:.3f} E={agg['Energy']:.0f}")
        rows.append(dict(model="MAPPO", **agg))

        # The deterministic (argmax) policy of the SAME learned actor, for
        # comparison.  This is not the policy Eq. (13) specifies, and the gap
        # between the two is a reported result, not an implementation detail.
        det = [eval_agent(agent, base, steps=base.episode_len, seed=s,
                           deterministic=True) for s in range(eval_seeds)]
        dagg = {k: float(np.mean([s[k] for s in det])) for k in det[0]}
        dstd = {k: float(np.std([s[k] for s in det])) for k in det[0]}
        print(f"{'MAPPO(argmax)':10s} CK={dagg['CK']:.3f}+-{dstd['CK']:.3f} "
              f"AG={dagg['AG']:.3f} SC={dagg['SC']:.3f} "
              f"CI={dagg['CovIntegral']:.3f}")
        rows.append(dict(model="MAPPO-argmax", **dagg))

    # ---- the achievability ceiling ---------------------------------
    # The teleporting placement oracle is the reference against which every
    # coverage number in this table is meant to be read.  It is a property
    # of the configuration, not of any controller, so it is reported once.
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from analysis.feasibility import analyse_scene
        ck = []
        for s in range(min(eval_seeds, 3)):
            r = analyse_scene(s, base, iters=60, restarts=8)
            ck.append(float(r["oracle_CK_connected"]))
        orc = float(np.mean(ck))
        print(f"{'Oracle':10s} CK={orc:.3f}  (teleporting placement, "
              f"ignores energy and motion)")
        rows.append(dict(model="Oracle (ceiling)", CK=orc, AG=1.0, SC=orc,
                         CovIntegral=0.0, Energy=0.0, Denied=0.0,
                         Reserve=0.0, Stranded=0.0, Returning=0.0))
    except Exception as exc:                      # pragma: no cover
        print(f"{'Oracle':10s} (not computed: {exc})")

    with open(out_csv, "w", newline="") as f:
        # the fieldnames are the UNION over the rows: the learned rows carry
        # the seed-to-seed spread and the seed count, and the rule baselines
        # do not, so taking the fields from the first row alone loses them
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    print(f"\nSaved -> {out_csv}")
    return agent, rows


def run_ablation(base: SimConfig, out_csv="results_ablation.csv", stress=True,
                 episodes=12, eval_seeds=3, seeds=1, only=None, append=False):
    """Turn off one proposed component at a time (Table VIII).

    stress=True  -- ablate on the S8 combined-stress scenario.  The stress
                    config is what actually exercises the reserve and
                    arbitration mechanisms, but K=2 coverage is not
                    achievable there, so the coverage numbers are bounded by
                    the geometry rather than by the policy.
    stress=False -- ablate directly on the base (feasible) configuration, so
                    the component contributions are measured on a task that a
                    controller can actually solve.  This is the table that
                    supports the paper's claims.

    PROTOCOL.  Every variant is trained on the same configuration with the
    same budget, from the same set of training seeds, and differs from the
    full configuration in exactly one component, so a difference is
    attributable to that component.  Each variant is trained `seeds` times
    and the row reports the mean over those training seeds together with the
    seed-to-seed spread of CK.  A single-seed ablation cannot support a
    claim either way: the seed-to-seed spread of the learner is larger than
    most of the effects being measured, so a one-seed table reports noise
    with the appearance of a result.

    The budget must also match the base comparison's, or the table measures
    the budget rather than the component.  `episodes` is therefore the same
    number the base comparison uses.

    `only` restricts the run to a named subset of variants (used to add a
    single missing row under the same protocol), and `append` adds the rows
    to an existing CSV instead of replacing it.
    """
    import copy, csv, json, os
    here = os.path.dirname(os.path.abspath(__file__))
    # the device this run trains on; it is part of the protocol, and a row
    # measured on another device is never reused
    RUN_DEVICE = os.environ.get("UAV_DEVICE", "cpu")
    analysis = os.path.join(os.path.dirname(here), "analysis")
    print(f"\n=== Ablations ({'S8 combined stress' if stress else 'feasible base config'}) ===")
    print(f" protocol: {episodes} episodes x {c_episode(base)} steps, "
          f"{seeds} training seeds, {eval_seeds} eval seeds")
    variants = {
        "Full": dict(),
        "no-criticality": dict(lam_C=0.0),
        "no-degraded": dict(lam_D=0.0),
        "no-energy-cost": dict(lam_E=0.0),
        "no-undercoverage": dict(lam_O=0.0),
        "no-reserve": dict(use_reserve=False),
        "no-arbitration": dict(use_charging_arbitration=False),
        "no-mask": dict(use_action_mask=False),
        "no-agent-id": dict(use_agent_id=False),
        "no-map": dict(map_grid=0),
        "no-pad-bearing": dict(use_pad_bearing=False),
        "VanillaMAPPO": dict(lam_C=0.0, lam_D=0.0, use_reserve=False,
                             use_charging_arbitration=False,
                             use_action_mask=False, use_agent_id=False,
                             map_grid=0, use_pad_bearing=False),
    }
    if only:
        variants = {k: v for k, v in variants.items() if k in only}
    existing = []
    if append and os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    # RESUME.  A row that was already measured under THIS protocol is
    # reused rather than trained again: the ablation takes hours, and an
    # interrupted run should not throw away the training it has done.  The
    # protocol is part of the row, so a row measured under a different
    # budget or a different number of seeds is never reused.
    stem = os.path.splitext(os.path.basename(out_csv))[0]
    done = {}
    jl = os.path.join(analysis, f"{stem}_rows.jsonl")
    if os.path.exists(jl):
        for line in open(jl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (d.get("out_csv") == os.path.basename(out_csv)
                    and bool(d.get("stress")) == bool(stress)
                    and int(d.get("episodes", -1)) == int(episodes)
                    and int(d.get("eval_seeds", -1)) == int(eval_seeds)
                    and int(d.get("seeds", -1)) == int(seeds)
                    # the device is part of the protocol: a variant
                    # trained on the GPU and its partner trained on the
                    # CPU are not a paired comparison
                    and str(d.get("device", "")) == RUN_DEVICE
                    and d.get("model") in variants):
                done[d["model"]] = d
    rows = []
    for name, kw in variants.items():
        if name in done:
            d = done[name]
            row = {k: v for k, v in d.items()
                   if k not in ("out_csv", "stress", "eval_seeds",
                                "seeds", "model", "per_seed_ck")}
            row["model"] = name
            # the budget is part of the protocol and belongs in the row:
            # it was dropped here, so a table of arms trained at a
            # different budget from the base comparison did not say so
            row["episodes"] = int(d.get("episodes", episodes))
            # a row measured before the device was recorded still comes
            # from this run's device, which the protocol pins
            row["device"] = str(row.get("device")
                                or os.environ.get("UAV_DEVICE", "cpu"))
            rows.append(row)
            print(f"  {name:18s} reused from a previous run "
                  f"(CK={row.get('CK', float('nan')):.3f})", flush=True)
            _write_ablation_csv(out_csv, existing + rows)
            continue
        per_seed = []
        for s in range(seeds):
            c = copy.deepcopy(base)
            if stress:
                # Ablate on the COMBINED-STRESS scenario (S8), not the base
                # config.  On the base config no UAV fails and nobody contends
                # for a pad, so the reserve / arbitration mechanisms never fire
                # and every ablation trivially matches.  The stress config is
                # what actually exercises them.
                c.failure_enabled = True
                c.failure_time = 500
                c.n_pads = 1
                c.B_max = 15_000.0
                c.charge_rate = 0.0004
                c.R_comm = 30.0
                c.R_cov = 30.0
            for k, v in kw.items():
                setattr(c, k, v)
            c.seed = s
            agent, _ = train_mappo(c, episodes=episodes, n_steps=c.episode_len,
                                   seed=s, verbose=False)
            # the device is recorded with the row, so that an artefact says
            # which device produced it: the floating-point kernels differ
            # between the CPU and the GPU, so two runs of the same
            # configuration on different devices are not the same
            # experiment
            device_used = str(agent.device)
            evs = [eval_agent(agent, c, steps=c.episode_len, seed=e)
                   for e in range(eval_seeds)]
            per_seed.append({k: float(np.mean([x[k] for x in evs]))
                             for k in evs[0]})
            print(f"  {name:16s} seed {s}: CK={per_seed[-1]['CK']:.3f} "
                  f"AG={per_seed[-1]['AG']:.3f} SC={per_seed[-1]['SC']:.3f} "
                  f"CI={per_seed[-1]['CovIntegral']:.3f}", flush=True)
        agg = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
        agg["Reserve"] = float(np.mean([p["Reserve"] for p in per_seed]))
        agg["Denied"] = float(np.mean([p["Denied"] for p in per_seed]))
        agg["Stranded"] = float(np.mean([p["Stranded"] for p in per_seed]))
        agg["CK_std"] = float(np.std([p["CK"] for p in per_seed]))
        agg["train_seeds"] = int(seeds)
        agg["episodes"] = int(episodes)
        # the per-seed values are kept, so that a variant can be compared
        # with the full configuration PAIRED: the variants share the
        # training seeds, so the difference is measured on the same
        # training runs and its spread is much smaller than the spread of
        # either row
        agg["per_seed_CK"] = ";".join(f"{p['CK']:.6f}" for p in per_seed)
        agg["device"] = device_used
        print(f"{name:18s} CK={agg['CK']:.3f}+-{agg['CK_std']:.3f} "
              f"AG={agg['AG']:.3f} SC={agg['SC']:.3f} "
              f"CI={agg['CovIntegral']:.3f} Reserve={agg['Reserve']:.1f} "
              f"Denied={agg['Denied']:.0f} Stranded={agg['Stranded']:.1f}",
              flush=True)
        rows.append(dict(model=name, **agg))
        # the row is written as soon as it is finished, so that an
        # interrupted run does not throw away the training it has done
        stem = os.path.splitext(os.path.basename(out_csv))[0]
        with open(os.path.join(analysis, f"{stem}_rows.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(_ablation_payload(
                rows[-1], out_csv, stress, episodes, eval_seeds, seeds,
                [p["CK"] for p in per_seed])) + "\n")
        # and the CSV is rewritten after every row, so that the artefact the
        # paper is built from is never behind the training
        _write_ablation_csv(out_csv, existing + rows)

    allrows = existing + rows
    _write_ablation_csv(out_csv, allrows)
    # the protocol is written next to the table, so that any row added later
    # (the extra "no-map" variant) is measured under the same budget
    with open(os.path.join(analysis, "ablation_protocol.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(out_csv=os.path.basename(out_csv), stress=stress,
                       episodes=int(episodes), eval_seeds=int(eval_seeds),
                       seeds=int(seeds)), f, indent=2)
    print(f"Saved -> {out_csv}")
    return rows


def _ablation_payload(row, out_csv, stress, episodes, eval_seeds, seeds,
                      per_seed_ck):
    """The incremental-log record of one finished ablation row.

    The payload is built by UPDATE and not by unpacking, because the row's
    own columns and the protocol's fields have overlapping names ("model",
    "episodes", "seeds"), and unpacking both into one dict() call raises
    "dict() got multiple values for keyword argument".  That crash killed
    the ablation stage once, so the construction is a function of its own
    and is tested.
    """
    payload = dict(row)
    payload.update(out_csv=os.path.basename(out_csv), stress=stress,
                   episodes=int(episodes), eval_seeds=int(eval_seeds),
                   seeds=int(seeds), per_seed_ck=list(per_seed_ck))
    return payload


def _write_ablation_csv(out_csv, rows):
    """Write the ablation CSV, preserving the column order of the rows."""
    import csv
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def c_episode(cfg):
    """The episode length of a configuration (a small helper for the logs)."""
    return int(getattr(cfg, "episode_len", 0))


def run_learned_baselines(base: SimConfig, out_csv="results_learned_baselines.csv",
                          episodes=12, eval_seeds=3, seeds=2):
    """The learned baselines of Table VI, evaluated on the same scenes.

    These are implementations of the METHOD FAMILIES the comparison table
    names -- a value-based multi-agent method and a PPO-family method that
    uses action masking -- not reproductions of the cited papers.  They use
    the same simulator, observation, action mask and team reward as the
    proposed method, so the only differences are the learning machinery.
    """
    import csv
    import copy
    from .baselines_rl import (train_id_dqn, train_ppo_local,
                                eval_id_dqn, eval_ppo_local)
    rows = []
    print("\n=== Learned baselines (Table VI) ===")

    for name, trainer, evaluator in [
            ("Independent DQN (value-based)", train_id_dqn, eval_id_dqn),
            ("Masked PPO, local critic", train_ppo_local, eval_ppo_local)]:
        cks, ags, scs, cis = [], [], [], []
        for s in range(seeds):
            c = copy.deepcopy(base)
            c.seed = s
            agent, _ = trainer(c, episodes=episodes, steps=c.episode_len,
                               seed=s, verbose=False)
            evs = [evaluator(agent, c, steps=c.episode_len, seed=e)
                   for e in range(eval_seeds)]
            agg = {k: float(np.mean([x[k] for x in evs])) for k in evs[0]}
            std = {k: float(np.std([x[k] for x in evs])) for k in evs[0]}
            cks.append(agg["CK"]); ags.append(agg["AG"]); scs.append(agg["SC"])
            cis.append(agg["CovIntegral"])
            print(f"  {name:32s} seed {s}: CK={agg['CK']:.3f}±{std['CK']:.3f} "
                  f"AG={agg['AG']:.3f} SC={agg['SC']:.3f} "
                  f"CI={agg['CovIntegral']:.3f}", flush=True)
        print(f"  {name:32s} MEAN     CK={np.mean(cks):.3f} "
              f"AG={np.mean(ags):.3f} SC={np.mean(scs):.3f} "
              f"CI={np.mean(cis):.3f}", flush=True)
        rows.append(dict(model=name, CK=float(np.mean(cks)),
                         CK_std=float(np.std(cks, ddof=1)) if len(cks) > 1
                         else 0.0,
                         # the protocol, so that the artefact says how it
                         # was measured: the number of training runs, the
                         # per-seed values they came from, and the device
                         train_seeds=int(len(cks)),
                         per_seed_CK=";".join(f"{v:.6f}" for v in cks),
                         device=str(getattr(agent, "device", "cpu")),
                         episodes=int(episodes),
                         eval_seeds=int(eval_seeds),
                         AG=float(np.mean(ags)),
                         SC=float(np.mean(scs)),
                         CovIntegral=float(np.mean(cis))))

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {out_csv}")
    return rows


def run_scenarios(base: SimConfig, out_csv="results_scenarios.csv",
                  episodes=12, seeds=1):
    """Table X: MAPPO vs the Reactive heuristic on every scenario variant.

    Each scenario defines two variants (e.g. sparse vs dense sensors). Both
    are evaluated; the agent is trained once on the base configuration and
    then transferred, which is the point of the exercise -- the policy is not
    retrained per scenario.
    """
    import csv
    import copy

    # The networks have fixed input sizes, and those sizes depend on the
    # configuration: the one-hot agent identity has length N, the global
    # state carries the sensor positions and the pad-occupancy vector, and
    # the actor's map summary is fixed.  A policy trained at one (N, M,
    # pads) combination therefore cannot be evaluated at another -- the
    # input layer does not fit.  One policy is therefore trained per
    # combination, and the scenarios that vary only the geometry of the
    # problem transfer within their combination.
    groups = {}
    for sid, question, cfgs in scenario_cfgs(base):
        for vi, c0 in enumerate(cfgs):
            key = (c0.N, c0.M, c0.n_pads)
            groups.setdefault(key, []).append((sid, vi, question, c0))
    agents = {}
    for key in sorted(groups):
        kN, kM, kp = key
        c = copy.deepcopy(base)
        c.N, c.M, c.n_pads = kN, kM, kp
        print(f"  training {seeds} policies for N={kN} M={kM} pads={kp} ...",
              flush=True)
        for s in range(seeds):
            ag, _ = train_mappo(c, episodes=episodes, n_steps=c.episode_len,
                                seed=s, verbose=False)
            agents.setdefault(key, []).append(ag)
            print(f"    seed {s} done", flush=True)

    rows = []
    print("\n=== Scenarios (MAPPO vs the rules) ===")
    for sid, question, cfgs in scenario_cfgs(base):
        for vi, c0 in enumerate(cfgs):
            c = copy.deepcopy(c0)
            # every policy of the group is evaluated on the SAME scene
            # (the evaluation is seeded from the scene's own seed), so the
            # comparison between the policies is paired
            evs = [eval_agent(ag, c, steps=c.episode_len, seed=c.seed)
                   for ag in agents[(c.N, c.M, c.n_pads)]]
            m = {k: float(np.mean([e[k] for e in evs])) for k in evs[0]}
            m["CK_std"] = (float(np.std([e["CK"] for e in evs], ddof=1))
                           if len(evs) > 1 else 0.0)
            m["train_seeds"] = len(evs)
            m["per_seed_CK"] = ";".join(f"{e['CK']:.6f}" for e in evs)
            # the device the policies were trained and evaluated on: it is
            # part of the protocol, and the row has to say which one
            m["device"] = str(getattr(
                agents[(c.N, c.M, c.n_pads)][0], "device", "cpu"))
            r = baseline_reactive(c, steps=c.episode_len, seed=c.seed)
            # the UNPRIVILEGED rule is the stronger of the two on the base
            # configuration, and the matrix has to be read against the
            # stronger reference
            rl = baseline_reactive_local(c, steps=c.episode_len, seed=c.seed)
            # a short description of the variant for the CSV
            variant = (f"N={c.N} M={c.M} Rcov={c.R_cov:.0f} "
                       f"Rcomm={c.R_comm:.0f} pads={c.n_pads} "
                       f"Bmax={c.B_max:.0f} fail={c.failure_enabled} "
                       f"clustered={c.start_clustered}")
            print(f"{sid}.{vi} ({question:16s}) {variant}  "
                  f"MAPPO CK={m['CK']:.3f} SC={m['SC']:.3f} "
                  f"CI={m['CovIntegral']:.3f} | "
                  f"Reactive CK={r['CK']:.3f} SC={r['SC']:.3f} | "
                  f"Reactive-local CK={rl['CK']:.3f}", flush=True)
            rows.append(dict(scenario=sid, variant=vi, question=question,
                             config=variant,
                             mappo_CK=m["CK"], mappo_SC=m["SC"],
                             mappo_AG=m["AG"],
                             mappo_CovIntegral=m["CovIntegral"],
                             # the protocol the row was measured under, and
                             # the per-training-seed values it came from:
                             # a row that does not say how it was measured
                             # cannot be checked
                             train_seeds=m.get("train_seeds"),
                             per_seed_CK=m.get("per_seed_CK"),
                             CK_std=m.get("CK_std"),
                             device=m.get("device"),
                             episodes=int(episodes),
                             react_CK=r["CK"], react_SC=r["SC"],
                             react_AG=r["AG"],
                             react_CovIntegral=r["CovIntegral"],
                             reactlocal_CK=rl["CK"], reactlocal_SC=rl["SC"],
                             reactlocal_AG=rl["AG"],
                             reactlocal_CovIntegral=rl["CovIntegral"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {out_csv}")
    return rows