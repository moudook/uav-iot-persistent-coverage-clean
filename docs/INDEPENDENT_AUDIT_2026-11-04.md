# Independent Novelty Audit - 2026-11-04 (subagent + main-agent verification)

Method: a clean-context explore subagent audited the report's 8 core claims
against live web literature. The main agent then INDEPENDENTLY verified the
subagent's findings via the arXiv API, OpenAlex, and direct fetches, and
attempted to counter them.

## Papers verified to exist (main agent, arXiv API direct)
- Babarytskyi, "Feasibility Ceilings Before Learning..." (arXiv:2610.20823, 2026-10-14) - yes
- MA-WAM (arXiv:2609.31281, 2026-09-25) - yes
- FAITH (arXiv:2610.12432, 2026-10-08) - yes
- PRIME (arXiv:2607.17922, 2026-07-20) - yes
- Fan et al., "Communication-Aware MARL..." (arXiv:2603.16141, 2026-03-17) - yes
- Shared Backbone PPO (arXiv:2605.17999, 2026-05-18) - yes
- Arribas et al., recharge scheduling (arXiv:2205.12656, 2022-05-25) - yes
- Ramesh & Thomas, fleet sizing (arXiv:2604.15890, 2026-04-17) - yes
- Shakhatreh et al., continuous coverage (arXiv:1705.09766, 2017) - yes
- Cetinkaya et al., energy-neutral coverage (arXiv:2306.12315, 2023) - yes
- POGS-QMIX, MDPI Drones 10(9):688 (2026) - yes, doi 10.3390/drones10090688

## POGS-QMIX content (verified via OpenAlex abstract)
Confirms: persistent multi-UAV coverage, finite battery + periodic
recharging, parameter-shared QMIX, return-energy action mask, greedy-vs-
Hungarian assignment ablation, multi-seed statistics (5 seeds),
energy-stress tests, operating-condition studies. It beats the greedy
baseline by ~12.6pp (high-priority coverage 0.547+/-0.009).
=> Our evaluation methodology (multi-seed, stress regimes, Hungarian
baseline) is pre-empted as "textbook" by a Sep 2026 paper.

## Fan et al. content (verified via arXiv abstract)
"approximately 12% increase in target coverage over MAPPO under restricted
communication", competitive with a MILP offline upper bound.
=> MAPPO is not a strong baseline in this domain; oracle/MILP ceilings are
already standard comparison practice in this line of work.

## Combination search (main agent, arXiv API)
- persistent+UAV+coverage+feasibility+energy -> only Arribas 2205.12656
- K-coverage+UAV+recharging+certificate -> none
- feasibility+ceiling+learning+UAV -> only Babarytskyi 2610.20823
No work found combining ALL of: K-redundant IoT-sensor coverage, PHY-gated
connectivity, charging contention, reserve/failure recovery, test-time
planning, and an energy-flow certificate. The combination is novel as a
package, but every component exists separately in near-identical form
within the last 4 months.

## Verdict (agrees between subagent and main-agent verification)
- NOT a breakthrough. Correct, honest, well-executed incremental/
  engineering contribution.
- The two headline "certificates" are elementary known results repackaged
  (packing lower bound = classic WSN k-coverage geometry; energy ceiling =
  conservation of energy, with formal fleet-sizing precedents).
- The learner's margin over rules is small (+0.028) vs POGS-QMIX's
  +12.6pp; mapPO itself is beaten by graph MARL (+12%).
- The genuinely good parts: honest negatives, corrected protocol, CIs,
  stress regimes, the assembled end-to-end protocol (feasibility -> oracle
  -> corrected eval -> planning -> distillation), and the energy certificate
  as the first non-vacuous ceiling at the reported config.
- Realistic venues: regional/mid-tier IEEE (NCC, COMSNETS workshops, ANTS,
  VTC/ICC workshops, MDPI Drones). Not top venues.

## Caveats
- The subagent's claim that POGS-QMIX uses "95% Student-t CIs" is a slight
  overstatement: abstract says "multi-seed statistics" (CIs plausible but
  not explicitly confirmed in the abstract).
- The layout-specific "13% of layouts exclude N=5" and exact 0.113/0.367
  energy ceilings are internally computed, not externally falsifiable.
