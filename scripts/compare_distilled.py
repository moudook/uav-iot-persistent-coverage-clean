"""Compare distilled (BC from planner) vs base learner vs planner oracle.

Reads:
  ml_impl/runs/exp_distilled/eval_heldout_2000.json
  ml_impl/runs/exp_ego_eval/eval_heldout.json        (base, 2000-step)
  ml_impl/runs/exp_ego_eval/eval_planning_2000.json  (planner oracle)
"""
import json, os, math

BASE = r"C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1"
RUNS = os.path.join(BASE, "ml_impl", "runs")


def load_heldout(tag, rel=None):
    p = os.path.join(RUNS, tag, rel or "eval_heldout.json")
    d = json.load(open(p, encoding="utf-8"))
    per = [v["mean"] for v in d["per_checkpoint"].values()]
    return per, d


def t_half(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n - 1, 2.0)
    return tcrit * sd / math.sqrt(n)


per_base, d_base = load_heldout("exp_ego_eval")
per_dist, d_dist = load_heldout("exp_distilled", "eval_heldout_2000.json")
per_pl, d_pl = load_heldout("exp_ego_eval", "eval_planning_2000.json")

mb = sum(per_base) / len(per_base)
md = sum(per_dist) / len(per_dist)
mp = sum(per_pl) / len(per_pl)

print("=== NOMINAL 2000-step held-out (20 scenes, 5 ckpts) ===")
print(f"base      : {mb:.4f}  CI [{mb - t_half(per_base):.4f}, {mb + t_half(per_base):.4f}]")
print(f"distilled : {md:.4f}  CI [{md - t_half(per_dist):.4f}, {md + t_half(per_dist):.4f}]")
print(f"planner   : {mp:.4f}  CI [{mp - t_half(per_pl):.4f}, {mp + t_half(per_pl):.4f}]")

d_dist_base = [x - y for x, y in zip(per_dist, per_base)]
mdb = sum(d_dist_base) / len(d_dist_base)
hdb = t_half(d_dist_base)
print(f"\npaired distilled-base : {mdb:+.4f}  95% CI [{mdb - hdb:+.4f}, {mdb + hdb:+.4f}]")
print("verdict: RESOLVED" if (hdb > 0 and abs(mdb) > hdb) else "verdict: not resolved")

d_pl_base = [x - y for x, y in zip(per_pl, per_base)]
mpb = sum(d_pl_base) / len(d_pl_base)
hpb = t_half(d_pl_base)
print(f"paired planner-base  : {mpb:+.4f}  95% CI [{mpb - hpb:+.4f}, {mpb + hpb:+.4f}]")

print("\nper-checkpoint:")
for k in d_base["per_checkpoint"]:
    b = d_base["per_checkpoint"][k]["mean"]
    d = d_dist["per_checkpoint"][k]["mean"]
    p = d_pl["per_checkpoint"][k]["mean"]
    print(f"  {k}: base={b:.4f} distilled={d:.4f} planner={p:.4f}  dist-base={d-b:+.4f}  pl-base={p-b:+.4f}")