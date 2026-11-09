"""Compare feasibility-aware planning (eval_planning_500.json) vs frozen base
policy (eval_ckpt_500.json) at the same 500-step protocol, same 20 scenes.

Reads:
  ml_impl/runs/exp_ego_eval/eval_planning_500.json
  ml_impl/runs/exp_ego_eval/eval_ckpt_500.json
"""
import json, os, math

BASE = r"C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1"
RUNS = os.path.join(BASE, "ml_impl", "runs", "exp_ego_eval")


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    per = [v["mean"] for v in d["per_checkpoint"].values()]
    return per, d


def t_half(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n - 1, 2.0)
    return tcrit * sd / math.sqrt(n)


per_base, d_base = load(os.path.join(RUNS, "eval_ckpt_500.json"))
per_pl, d_pl = load(os.path.join(RUNS, "eval_planning_500.json"))

mb = sum(per_base) / len(per_base)
mp = sum(per_pl) / len(per_pl)
print("=== PLANNING vs BASE at 500-step protocol (20 scenes, 5 ckpts) ===")
print(f"base    : {mb:.4f}  CI [{mb - t_half(per_base):.4f}, {mb + t_half(per_base):.4f}]")
print(f"planned : {mp:.4f}  CI [{mp - t_half(per_pl):.4f}, {mp + t_half(per_pl):.4f}]")
diffs = [p - b for p, b in zip(per_pl, per_base)]
dm = sum(diffs) / len(diffs)
dh = t_half(diffs)
print(f"paired planned-base: {dm:+.4f}  95% CI [{dm - dh:+.4f}, {dm + dh:+.4f}]")
if dh > 0 and abs(dm) > dh:
    print("verdict: RESOLVED")
else:
    print("verdict: not resolved")
print()
print("per-checkpoint:")
for k in d_base["per_checkpoint"]:
    b = d_base["per_checkpoint"][k]["mean"]
    p = d_pl["per_checkpoint"][k]["mean"]
    print(f"  {k}: base={b:.4f} planned={p:.4f} diff={p-b:+.4f}")
print()
print("H=", d_pl.get("H"), "K=", d_pl.get("K"), "steps=", d_pl.get("steps"))