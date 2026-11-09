# INNOVATION RESULTS - 2026-11-04 session (part 2)

Follow-up to the 2026-11-03 session, driven by SI-3 + SI-4 feedback.

## Solved: the CTDE violation in Innovation 10 (planning)
- The planner was relabelled in the report as a centralised test-time oracle
  (it clones the global simulator and searches joint actions). Not a
  deployable controller.
- New: planner distillation. `scripts/gen_planning_dataset.py` collects
  (obs, action) pairs from the planner (H=2, K=4, 500 steps, 10 scenes x
  5 ckpts = 25,000 joint steps = 200,000 per-agent samples).
  `scripts/distill_actor.py` fine-tunes each checkpoint's actor by BC
  (5 epochs, CE loss) -> `exp_distilled` checkpoints, decentralised,
  zero global state at deployment. Evaluation at 2000 steps pending.

## Classical OR baseline (answers the "weak baseline" criticism)
- `scripts/eval_classical_baseline.py`: oracle-formation (search_best
  connected) + Hungarian assignment per step + fixed charging rotation.
- Nominal 2000-step, 20 scenes: 0.929 (sd 0.062) vs learner 0.9512,
  reactive rule 0.923. The classical scheduler does NOT beat MAPPO.
- Stress 2000-step (20 scenes):
    nominal       0.929   small_battery 0.791
    single_pad   0.901   low_comm      0.808
    clustered     0.980   failure       0.919
    multi_failure 0.919
  Learner beats classical on every same-architecture regime. Classical
  shares the small_battery deficit (0.791 vs rule 0.829), beating the
  learner (0.765) but still losing to the explicit rule.
- Added classical column to report stress table (tab:stress) + prose.

## Energy certificate (SI-4 A) - built 2026-11-04
- `scripts/energy_certificate.py`: steady-state power-balance bound
  (S*P_pad_net/P_air), finite-horizon airborne-seconds bound, energy-aware
  LP ceiling (fluid relaxation, scipy linprog).
- Result: at N=8, S=2, B=30 kJ, 30 W pads, 55.1 W airborne:
    steady-state max airborne = 0.904 UAVs -> persistent ceiling 0.113
    T=400s: ceiling 1.000 (feasible but draws down packs)
    T=2000s: ceiling 0.367 (full) / 0.326 (mean charge 0.85)
  Only N=12, S=4, B=60 kJ, cr=0.0004 is persistently feasible.
- Report subsection `sec:energyflow` added. Caveat on framing: fleet-sizing
  conservation laws exist (Arribas 2023, Ramesh-Thomas 2026), the power-flow
  + finite-horizon + LP ceiling combination appears unpublished (web audit).
  Report cites both + Song 2024 (K-coverage energy MILP).

## Web audit (2026-11-04, subagent via arXiv/OpenAlex)
- Verified all cited 2025-2026 works exist (MA-WAM 2609.31281,
  Babarytskyi 2610.20823, FAITH 2610.12432, POGS-QMIX Drones 10(9):688,
  PRIME 2607.17922, Fan 2603.16141, Shared Backbone PPO 2605.17999).
- Energy certificate package (3 parts) not found published as a unit;
  component (1) is repackaged conservation law -> framed as such in report.

## Failed arms (dropped)
- Decentralised 1-hop lookahead heuristic (eval_decentral_lookahead.py):
  naive local scoring (0.816 @200 steps vs base 0.885) - the local
  kinematic filter does not capture the planner's gain. Dropped; the
  distillation route is the path (it moves the gain INTO the policy).

## Status
- Report: 27 pages, 0 LaTeX errors, dead-number check clean. Commits:
  9979cd1 (energy + reframe), 20e43d2 (classical + distill pipeline),
  d4c4ae9 (stress table + classical column).
- Pending: exp_distilled eval result + compare_distilled.py verdict.
