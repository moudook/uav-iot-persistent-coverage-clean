# DEMO GUIDE — Progress-Meeting Demo for Prof. (updated for clean report, 25 pages)

Purpose: show what the project actually is now. Report is self-contained; no history section remains.

The story in one line:
> "We built two things: a feasibility certificate that says whether a coverage number is meaningful, and a learned controller evaluated honestly under it. The certificate is the contribution."

Do not say "learning wins" universally. Say: learner 0.9512 vs strongest rule 0.923 (+0.028) on nominal; learning wins on low connectivity (+0.125) and failures; loses on small battery and spread start; the config is not persistently energy-feasible (ceiling 0.113).

Total time: 10-12 minutes if followed tightly.

---

## 0.5 PRE-FLIGHT CHECK (15 minutes before)

| Window | Purpose |
|---|---|
| PowerShell in repo | `cd C:\Users\gbust\Downloads\Academic\Courses\Machine_learning_in_wireless_network\Project1` |
| PDF viewer | `PPT_SI_CHANGES.pdf` and `Mlawn-midsem-tex.pdf` ready |

WSL status (re-verified 2026-11-09):
WSL distro exists (Ubuntu). After enabling Virtual Machine Platform and reboot, wsl -d Ubuntu -- echo WSL OK passes and the Cmdenv run reaches t=40s, event #71885, End.

Windows-native checks that DO work (tested today):

```powershell
python scripts/energy_certificate.py --T 400
```
Expected: `persistent CK ceiling = 0.113   feasible = False`, finite-horizon ceiling `1.000`. This takes seconds.

```powershell
python ml_impl/analysis/claim_audit.py
```
Expected tail: `34 claims verified, 0 FAILED, 4 recorded for transparency`.

If either of these fail, stop and fix before the meeting.

---

## DEMO SCRIPT (10-12 minutes)

### A. Current project in one slide (2 min)

Open `PPT_SI_CHANGES.pdf`:

1. Report: Energy-Flow Feasibility... title, 25 pages, feasibility-certified study.
2. Headline: Learner 0.9512, strongest rule 0.923, classical OR scheduler 0.929, oracle ceiling 1.000.
3. Split: wins on low connectivity and failures; loses on small battery (-0.064) and spread start (-0.038).
4. Energy: 0.90 sustainable airborne UAVs vs 8 needed → ceiling 0.113. 400 s drains packs; not persistent.

Say: "The contribution is not RL always beats heuristics; it is a certificate and protocol that says exactly when it does and does not."

### B. Open the report (2 min)

Open `Mlawn-midsem-tex.pdf`:

1. Evaluation protocol: frozen policy, held-out scenes (seeds 1000+), paired intervals.
2. Stress table: see the split above.
3. Energy-Flow Feasibility: the energy certificate and ceiling.
4. No appendix. No correction history. The paper stands on its current numbers only.

### C. Simulation — Python side live (3 min, no WSL needed)

Run live in PowerShell:

```powershell
python scripts/energy_certificate.py --T 400
```
Point at: `persistent CK ceiling = 0.113`, `feasible = False`. Then `--T 2000` shows ceiling 0.367.

Then:

```powershell
python scripts/energy_certificate.py --T 400 --sweep
```
Point at: only the combined config (N=12, 4 pads, 60 kJ, 0.0004) reaches ceiling 1.000.

### D. Simulation — OMNeT++/INET (3 min)

Live run PASSES today (2026-11-09, after enabling Virtual Machine Platform and reboot):

1. `Simulation/omnet_uav/UavNetwork.ned` — network: `UavNetwork`, `numUavs=8`, `numSensors=20`, BonnMotion mobility.
2. `Simulation/omnet_uav/omnetpp.ini` — `sim-time-limit=40s`, trace `episode_000.movements`, 0.5 W, 6 Mbps, UdpBasicApp.
3. Recorded evidence: `Simulation/omnet_uav/results/General-#0.sca` holds `packets sent` 40 / `packets received` per UAV, scalars recorded at `t=40 s, event #71885, End` (verified 2026-11-04 when WSL was available; backups at `omnetpp.ini.bak-20261104` in the WSL example dir).

What to say:
> "This is INET, not the Python simulator. The BonnMotion trace comes from the Python side; INET models 802.11 MAC/PHY and writes packets sent/received scalars. The co-simulation is the fidelity check."

For the visual window, run Qtenv (opens a real GUI window on Windows via WSLg):

```powershell
wsl -d Ubuntu -- bash -c "cd /home/gbust/omnetpp_ws && DISPLAY=:0 timeout 60 opp_env run inet-4.6.0 -c 'cd /home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet && opp_run -u Qtenv -l /home/gbust/omnetpp_ws/inet-4.6.0/out/clang-release/src/libINET.so -n /home/gbust/omnetpp_ws/inet-4.6.0/src:/home/gbust/omnetpp_ws/inet-4.6.0/examples /home/gbust/omnetpp_ws/inet-4.6.0/examples/uav_omnet/omnetpp.ini' 2>&1 | tail -5"
```

Verified 2026-11-09: NED files load, images load, window appears. Press the express/fast-run button in Qtenv to finish in seconds. Blue nodes are UAVs replaying our traces; grey nodes are the 20 sensors; arrows are 802.11 frames; the red circle is the sink.

If the Qtenv window does not appear, the machine needs WSLg display support; fall back to the Cmdenv headless run above and say so explicitly.

### E. Audit (2 min)

Run `python ml_impl/analysis/claim_audit.py` and show the tail: `34 pass, 0 fail`. PHY equivalence 0.012 dB over 24 points. Determinism 170/170.

Note: the audit covers physical/feasibility/protocol claims, not the tuned result table. Say so.

### F. Close (1 min)

> "The project gives a certificate, a frozen-policy comparison, and a clear answer on where learning helps. The value is being honest about the boundary."

---

## CHEAT SHEET (quote the report, not memory)

| Fact | Value |
|---|---|
| Frozen learner | 0.9512 |
| Strongest hand rule | 0.923 |
| Difference | +0.028, interval excludes 0 |
| Low-comm gain | +0.125 |
| Classical OR scheduler | 0.929 |
| Planning oracle | 0.9723 (not CTDE; global) |
| Distilled actor | 0.9618 (+0.0106, directional) |
| Energy steady-state ceiling | 0.113; config not persistent |
| Claim audit | 38 items; 0 failing |
| Determinism | 170 / 170 |

---

## FREQUENT QUESTIONS

1. **Is this actually OMNeT++?** Yes, INET 4.6 / OMNeT++ 6.4 was verified at `t=40 s, event #71885, End` with packet scalars in `General-#0.sca`. Live WSL launch verified 2026-11-09 after enabling Virtual Machine Platform
2. **Why learning wins sometimes?** It does not win universally. It wins on low connectivity and faults; loses where energy/start geometry dominate. The certificate is what makes the split visible.
3. **Can we call this persistent?** No. At 400 s the swarm spends its initial packs. State the horizon explicitly.
4. **What is the new contribution?** A feasibility certificate (spatial packing + energy flow) plus corrected held-out evaluation plus honest negatives.
5. **What is not supported?** Long-horizon persistence, guaranteed superiority over all heuristics, deployable CTDE planning without distillation, hardware validation.
6. **One chart?** Stress regime table (PPT slide).

---

## FALLBACK PLAN

| Failure | Use instead |
|---|---|
| WSL launch fails | Show `PPT_SI_CHANGES.pdf` + `Mlawn-midsem-tex.pdf` + run the two Python commands in Sec 0.5. Say: "Recorded artefacts are on disk; live INET needs virt platform enabled." |
| Python checks fail | Stop. Show only the PDFs. Do not repair live. |
| PDF not ready | Show source paths; do not invent a page number. |

---

## FILES TO HAVE OPEN

- `PPT_SI_CHANGES.pdf`
- `Mlawn-midsem-tex.pdf`
- `Simulation/omnet_uav/` (NED, ini, traces)
- `ml_impl/` (code)
- `DEMO_GUIDE.md` (this file)

## SESSION RECORD

- 2026-11-04: WSL INET example aligned to repo config; run reached `t=40 s, event #71885, End`; scalars regenerated in `General-#0.sca`.
- 2026-11-09: WSL launch failed with `HCS_E_HYPERV_NOT_INSTALLED` until Virtual Machine Platform was enabled with `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart` and a reboot. After reboot, live Cmdenv passed (t=40s, event #71885, End) and Qtenv loaded (NED + images) and showed the window. Windows-native checks (energy certificate, claim audit) re-verified today and passing.
- Report is now 25 pages, self-contained, no correction history. Deck is 16 slides, last four planning slides removed.
