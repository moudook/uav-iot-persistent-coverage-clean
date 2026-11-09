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

pl = json.load(open(os.path.join(RUNS, "eval_planning_2000.json"), encoding="utf-8"))
base = json.load(open(os.path.join(RUNS, "eval_heldout.json"), encoding="utf-8"))

per_pl = [v["mean"] for v in pl["per_checkpoint"].values()]
per_base = [v["mean"] for v in base["per_checkpoint"].values()]

m_pl = sum(per_pl) / len(per_pl)
m_base = sum(per_base) / len(per_base)
print("=== PLANNING vs BASE at 2000-step headline protocol (20 scenes, 5 ckpts) ===")
print(f"base    : {m_base:.4f}  CI [{m_base - t_half(per_base):.4f}, {m_base + t_half(per_base):.4f}]")
print(f"planned : {m_pl:.4f}  CI [{m_pl - t_half(per_pl):.4f}, {m_pl + t_half(per_pl):.4f}]")

diffs = [p - b for p, b in zip(per_pl, per_base)]
dm = sum(diffs) / len(diffs)
dh = t_half(diffs)
print(f"paired planned-base: {dm:+.4f}  95% CI [{dm - dh:+.4f}, {dm + dh:+.4f}]")
verdict = "RESOLVED" if (dh > 0 and abs(dm) > dh) else "not resolved"
print("verdict:", verdict)
print()
print("per-checkpoint:")
for k in base["per_checkpoint"]:
    b = base["per_checkpoint"][k]["mean"]
    p = pl["per_checkpoint"][k]["mean"]
    print(f"  {k}: base={b:.4f} planned={p:.4f} diff={p-b:+.4f}")

# rule comparison
rule = 0.9169  # Reactive-local at 500 steps; at 2000 it's 0.923
print()
print("best rule (Reactive-local) at 2000 steps = 0.923 (report) / at 500 = 0.9169")
print(f"planned - rule (2000 vs 0.923) = {m_pl - 0.923:+.4f}")
print(f"base - rule (2000 vs 0.923) = {m_base - 0.923:+.4f}")

# headroom gate: ceiling - best rule
print()
print(f"headroom (ceiling 1.0 - rule 0.923) = {1.0 - 0.923:.4f}")
print(f"headroom against planned = {1.0 - m_pl:.4f}")