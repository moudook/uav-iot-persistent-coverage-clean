"""Compare feasibility-aware planning vs base across stress regimes at the
same step protocol (both at 500 steps, same scenes).

Reads:
  ml_impl/runs/exp_ego_eval/eval_planning_stress.json
  ml_impl/runs/exp_ego_eval/eval_ckpt_stress_base.json
"""
import json, os, math

BASE = r"C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1"
RUNS = os.path.join(BASE, "ml_impl", "runs", "exp_ego_eval")

def t_half(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n-1, 2.0)
    return tcrit * sd / math.sqrt(n)

pl = json.load(open(os.path.join(RUNS, "eval_planning_stress.json"), encoding="utf-8"))
bs = json.load(open(os.path.join(RUNS, "eval_ckpt_stress_base.json"), encoding="utf-8"))

print("=== PLANNING vs BASE by stress regime (%d steps, %d scenes, 5 ckpts) ==="
      % (pl["steps"], len(pl["scenes"])))
print(f"{'regime':14s} {'base':>8s} {'planned':>8s} {'diff':>8s} {'95% CI':>22s}  verdict")
for reg in ["nominal", "small_battery", "low_comm", "clustered", "failure", "multi_failure"]:
    if reg not in pl.get("regimes", {}) or reg not in bs.get("regimes", {}):
        continue
    p = pl["regimes"][reg]["per_seed_planning"]
    b = bs["regimes"][reg]["per_seed_base"]
    n = min(len(p), len(b))
    diffs = [x - y for x, y in zip(p[:n], b[:n])]
    dm = sum(diffs) / n
    dh = t_half(diffs)
    verdict = "RESOLVED" if (dh > 0 and abs(dm) > dh) else "not resolved"
    print(f"{reg:14s} {sum(b)/n:.4f} {sum(p)/n:.4f} {dm:+.4f} [{dm-dh:+.4f}, {dm+dh:+.4f}]  {verdict}")