# DEMO GUIDE — Progress-Meeting Demo for Prof. (verified 4 Nov 2026)

Purpose: show the professor what the project actually is now, not the old midsem claim set.

The story in one line:
> “We fixed a broken measurement, built a feasibility/evaluaton framework that shows where learning helps and where it does not, and implemented an honest architecture that is now a corrected comparison rather than a claim that learning wins everywhere.”

This demo must not say "learning wins" or "MAPPO beats the rule" as a universal claim. It must say:
- On the nominal channel, learner 0.9512 vs strongest rule 0.923 (+0.028)
- Learning wins clearly on low connectivity (+0.125) and both failure regimes; loses on small battery and spread start
- The configuration is not persistently energy-feasible beyond 400 s
- Classical OR scheduler scores 0.929 below the learner
- The certificate and evaluation protocol are the real contribution

Total time: 10–12 minutes if followed tightly.

---

## 0.5) PRE-FLIGHT CHECK (complete these 15 minutes before the demo)

Open three windows on Windows:

| Window | Purpose |
|---|---|
| Windows PowerShell in repo | `cd C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1` |
| WSL Ubuntu | `wsl -d Ubuntu` |
| PDF viewer | `PPT_SI_CHANGES.pdf` and `Mlawn-midsem-tex.pdf` ready to show |

Run the one-minute OMNeT++ sanity check:

```powershell
wsl -d Ubuntu -- bash -c "cd /home/gbust/omnetpp_ws && opp_env run inet-4.6.0 -c 'cd /home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet && opp_run -u Cmdenv -l /home/gbust/omnetpp_ws/inet-4.6.0/out/clang-release/src/libINET.so -n /home/gbust/omnetpp_ws/inet-4.6.0/src:/home/gbust/omnetpp_ws/inet-4.6.0/examples /home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet/omnetpp.ini 2>&1 | tail -12'"
```

Required output (verified today):

```text
** Event #71885   t=40   ... 100% completed ...
Simulation time limit reached -- at t=40s, event #71885
End.
```

If this fails, go to §8. Do not claim it passed.

Then run this scalar check:

```powershell
wsl -d Ubuntu -- bash -c "cd /home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet && grep -E 'scalar.*(packets sent|received)' results/General-#0.sca | head -16"
```

Required output: `scalar UavNetwork.uav[i].app[0] "packets sent" 40` and `"packets received"` values with indices 0–7.

Important: before today this exact example failed with a `NoSuchMobility` configuration error. The repo copy and WSL example now match: `UavNetwork`, 8 UAVs, 20 sensors, `BonnMotionMobility` mobility, 40 s sim-time. The WSL INET example was aligned to the repo copy and the old mismatch was backed up as `omnetpp.ini.bak-20261104`.

---

## DEMO SCRIPT (10–12 minutes)

### A. The current project, in one slide (2 min)

Open `PPT_SI_CHANGES.pdf` and show:

1. **What the report is now**
   - Title: "Energy-Flow Feasibility and Learned Redundancy Management for Persistent K-Coverage UAV Networks"
   - 27 pages; feasibility-certified evaluation; appendix holds the bug-retraction story.
   - The evaluated observation is 68-dimensional: 30 base + 16 map grid + 8 agent ID + 3 pad bearing + 11 ego-centric deficit/repulsion.

2. **Headline numbers** (all held-out scenes; frozen policy; paired intervals)
   - Learner: 0.9512
   - Strongest rule: 0.923
   - Classical OR scheduler: 0.929
   - Oracle ceiling: 1.000
   - Bug-fixed, pre-redesign: 0.859

3. **Where learning wins and loses**
   - Nominal: +0.026 over rule
   - Low connectivity: +0.125 over rule
   - Failure: +0.034; multi-failure: +0.035
   - Small battery: -0.064 vs rule
   - Spread start: -0.038 vs rule

4. **The honest energy finding**
   - 2 pads deliver 24.9 W net each; one airborne UAV draws 55.1 W.
   - Steady state: 0.90 UAVs sustainable, versus 8 needed -> persistent ceiling 0.113.
   - At 400 s: ceiling 1.000 (only by draining the initial packs).
   - At 2000 s: ceiling 0.367 (full) / 0.326 (mean initial charge).
   - Only a much larger configuration reaches a 1.000 ceiling.

Say: "The project's contribution is not 'RL always beats heuristics'; it is a certificate and protocol that shows exactly when it does and does not."

### B. Open the current report (2 min)

Open `Mlawn-midsem-tex.pdf` and show:

1. **Section VI / Evaluation protocol** — frozen policy, fixed held-out scenes, paired comparisons.
2. **Section VIII-E / Stress regimes** — the table above.
3. **Section III-H / Energy-Flow Feasibility** — the energy certificate and its limit.
4. **Appendix** — the earlier measurement bug and the correction.

If the professor asks "what broke?" answer:
> “The first version used a sign-flipped entropy bonus and ran one gradient step per episode. The old 0.298 gap was a broken trainer plus an unfair comparison. That entire result was retracted and is now in the appendix."

### C. Show OMNeT++/INET co-simulation (3 min)

Do not open an untested GUI and hope. Use the tested Cmdenv path:

1. Run (or re-run) the WSL command from §0.5.
2. Show the final lines:
   - `event #71885`
   - `Simulation time limit reached -- at t=40s`
   - `End.`
3. Show the scalar file with:
   ```powershell
   wsl -d Ubuntu -- bash -c "cd /home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet && grep -E 'scalar.*(packets sent|received)' results/General-#0.sca | head -16"
   ```
4. Show the files in the example path that the simulation used:
   - `/home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet/UavNetwork.ned`
   - `/home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet/omnetpp.ini`
   - `traces/episode_000.movements`

What to say:
> “This is INET, not the Python simulator. The BonnMotion trace comes from the Python side; INET models 802.11 MAC/PHY and records packets sent/received in `General-#0.sca`. The co-simulation is the fidelity check, not a claim that it shares identical topology as the test suite.”

### D. Show the audit (2 min)

Use the report/claims audit:

- Claim audit: **38 items; 34 pass; 4 informational; 0 fail**
- Determinism: **170 / 170**
- PHY algebraic equivalence: **0.012 dB over 24 points**
- Note: the audit covers physical/feasibility/protocol claims, not the tuned result table.

If asked about accuracy:
> “The audit recomputes the claims we can derive exactly. It does not vouch for the tuned experimental values, which are artifacts of the runs, and those are the part we keep transparent.”

### E. Show the current report quickly (1 min)

At the end, go back to two lines:
> “The project gives a certificate, a corrected comparison, and a clear answer on where learning helps.”
> “It does not show an intractable win anywhere; the value is in being honest about the boundary.”

---

## CHEAT SHEET (quote from the report, not from memory)

| Fact | Value |
|---|---|
| Frozen learner | 0.9512 |
| Strongest hand rule | 0.923 |
| Difference | +0.028, 95% interval excludes 0 |
| Low-comm gain | +0.125 |
| Classical OR scheduler | 0.929 |
| Centralised planning oracle | 0.9723 (not CTDE; global) |
| Deployable distilled actor | 0.9618 (paired +0.0106, not resolved at this n) |
| Energy certificate (steady state) | 0.113 ceiling; config not persistent |
| Persistent feasibility | only much larger config |
| Claim audit | 38 items; 0 failing |
| Determinism | 170 / 170 |

---

## LIKELY QUESTIONS AND HONEST ANSWERS

1. **Is this actually OMNeT++?**
   - Yes. Cmdenv run completed above; the log shows t=40 s, event #71885; INET packet-level 802.11 is in the loop. PyTorch does not replace INET.

2. **Does the simulator show our results?**
   - The simulator produces network-layer observables; these feed the physical-layer validation portion of the report, not a new proof of optimal control.

3. **Is the project using our own code?**
   - Yes: `Simulation/`, `ml_impl/`, `scripts/`. The WSL copy mirrors `Simulation/omnet_uav/` after the 20261104 sync.

4. **Why is learning winning?**
   - Not established as a universal fact. It wins on low connectivity and judged failures; loses on battery and spread start. The honest rate is “more than the rule on this set”, not “RL wins”.

5. **Why did the earlier version claim the opposite?**
   - The old result came from a broken training setup and a train/test mismatch. It was retracted; details are in the appendix, not reused.

6. **Can we say this is a practical deployment?**
   - No. The benchmark is a decision-time test over 400 s with a fixed configuration. It does not prove a persistent mission or hardware readiness.

7. **What exactly is the new contribution?**
   - A feasibility certificate (spatial packing + energy flow + LP ceiling), plus corrected held-out evaluation, plus honest negatives.

8. **What is not supported?**
   - Long-horizon persistence; guaranteed superiority over all heuristics; deployable CTDE planning without distillation; hardware validation.

9. **If the professor asks for one chart**
   - Show `PPT_SI_CHANGES.pdf`, slide "Stress regimes".

---

## FALLBACK PLAN (if something fails)

Do not attempt repairs or cosmetic changes during the demo.

| Failure | Use instead |
|---|---|
| WSL command fails | Show `PPT_SI_CHANGES.pdf` and `Mlawn-midsem-tex.pdf` only; run nothing live; say: "I can show the recorded artefacts; the current config reaches t=40 s, event #71885, End." |
| Cmdenv is slow | Show the verified log output already captured in this file. |
| INET scalar output is empty | Show `Simulation/omnet_uav/omnetpp.ini` in the repo and narratediv simply: "the config is the intended one; the results file is present under `results/General-#0.sca`." |
| PDF not ready | Show the report source path `Mlawn-midsem-tex.tex` and summarise; do not fabricate a page number. |

If you decide to pre-record the demo, stop playing with the live INET run; use the recorded failure as evidence that we tested it, not as evidence of success.

---

## FILES TO HAVE OPEN

- `C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1\PPT_SI_CHANGES.pdf`
- `C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1\Mlawn-midsem-tex.pdf`
- `C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1\Simulation\omnet_uav\`
- `C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1\ml_impl\`
- `C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1\DEMO_GUIDE.md`

WSL example path:

`/home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet`

---

## SESSION RECORD (for the meeting minute)

- 2026-11-04: The WSL INET example had a mismatching two-UAV `NoSuchMobility` configuration and failed before network setup.
- The example was aligned to `Simulation/omnet_uav` in the repo; backups for `omnetpp.ini` and `UavNetwork.ned` were saved in the WSL example directory as `*.bak-20261104`.
- The verification run completed:
  - `event #71885`
  - `t = 40 s`
  - `End.`
- `results/General-#0.sca` was regenerated; scalar entries list packets sent/received, with 8 UAVs.
- `DEMO_GUIDE.md` was rewritten; old version kept as `DEMO_GUIDE.backup-20261104.md`.