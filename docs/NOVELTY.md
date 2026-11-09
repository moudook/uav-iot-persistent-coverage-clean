# NOVELTY — novelty gate for C1–C3

Date: 2026-10-25. Searches: web search (Qwen/DeepSeek-equivalent via websearch). Primary sources fetched where possible.

## C1 — Network-in-the-loop evaluation (OMNeT++/INET co-sim where policies act only on received packets)

Closest papers:
1. FlyNetSim (Baidya & Shaikh, 2018, ACM MobiCom workshop / arXiv 1808.04967) — ns-3 + ArduPilot co-sim for UAV networks. Does NOT run MARL policies in the loop; no RL control. URL: https://arxiv.org/abs/1808.04967
2. AVENS (OMNeT++ + X-Plane) — OMNeT++/INET + flight simulator co-sim via XML. Does NOT run MARL policies; single UAV focus. URL: https://omnetpp.org/download-items/AVENS.html
3. CORNET (IISc) — NS-3 co-simulation middleware for robots with time sync. Generic, not MARL-specific, not UAV-coverage. URL: https://ece.iisc.ac.in/~htyagi/papers/comsnet-20.pdf
4. MARSNet (BU) — ns-3 + ROS multi-UAV sim for demand-response protocol. Not MARL control loop; not K-coverage. URL: https://cs-people.bu.edu/rmancuso/files/papers/MARSNet_AIAA19.pdf
5. CUSCUS — FL-AIR + NS-3 via tap bridges. Limited extensibility (no D2D/LTE/5G).

Delta: None of these run a MARL policy in a closed loop over a realistic 802.11 ad-hoc network for persistent K-coverage. Our C1 measures sensor->sink PDR/delay as a first-class metric alongside K-coverage, with the policy acting only on received packets. NOVEL (gap confirmed).

## C2 — Topology-conditioned information model + training

Closest papers:
1. CALF (arXiv 2603.12543, 2026) — communication-aware RL under delay and loss. Uses i.i.d. degradation, NOT topology-conditioned. Delta: our channel is topology-dependent (distance, hop count, neighbour count/load, topology state).
2. TAG-MAPPO (arXiv 2602.10029v2, 2026) — Topology-Aware Graph MAPPO for UAV coverage under node failure. Uses graph features, but NOT network-in-the-loop (no realistic packet delivery); NOT persistent K-coverage with charging. Delta: our information model is fitted from actual INET packet traces, not graph features.
3. MRLMN (arXiv 2505.08448, 2025) — multi-hop networking MARL with LLMs. For network coverage, not K-coverage of sensors; not topology-conditioned training.
4. Sun et al. 2023 (Neurocomputing) — difference/WLU rewards for persistent UAV coverage. Age-based, 1-coverage, no connectivity. Delta: our credit includes connectivity criticality and delivery criticality.
5. POGS-QMIX (Drones 2026) — persistent coverage, energy, recharging, return-energy mask, greedy ablation. Centralized online coordination, no failure recovery/reserve, no network-in-the-loop.

Delta: No paper fits a topology-conditioned information channel from realistic 802.11 traces and trains policies under it for persistent K-coverage. NOVEL (gap confirmed, but TAG-MAPPO is the closest competitor — must differentiate clearly).

## C3 — Redundancy- and criticality-aware credit

Closest papers:
1. Sun et al. 2023 (WLU, difference reward) — age-based, 1-coverage, no connectivity. Delta: our credit has K-coverage marginal + connectivity criticality (union-find) + delivery criticality.
2. TAG-MAPPO (arXiv 2602.10029v2) — graph-based feature, but credit is not decomposed into K-coverage marginal / connectivity criticality / delivery criticality.
3. Submodular MARL theory (arXiv 2603.06810) — theoretical framework; our min(n_d/K,1) objective is monotone submodular, connect to this.
4. COMA (Foerster et al., AAMAS 2018) — counterfactual baseline for credit. Our closed-form signals are cheaper (no critic evaluation per agent).
5. POGS-QMIX (Drones 2026) — QMIX value factorization for persistent coverage; no explicit counterfactual credit decomposition.

Delta: Closed-form per-agent counterfactual signals (K-coverage marginal, connectivity criticality, delivery criticality) for persistent K-coverage with charging and failure. Distinguish from Sun 2023's WLU. NOVEL (gap confirmed).

## Risk assessment
- TAG-MAPPO is the closest competitor for topology-awareness. Must clearly state our delta: (a) network-in-the-loop with actual 802.11 packet delivery, not graph features; (b) persistent K-coverage with charging/failure/reserve; (c) topology-conditioned training (not just topology-aware features).
- CALF is the closest for communication-aware training. Must state delta: topology-conditioned (not i.i.d.) channel.
- POGS-QMIX is the closest for persistent coverage MARL. Must state delta: failure recovery/reserve, network-in-the-loop, credit decomposition.

## Sources (verified URLs)
- FlyNetSim: https://arxiv.org/abs/1808.04967
- AVENS: https://omnetpp.org/download-items/AVENS.html
- TAG-MAPPO: https://arxiv.org/html/2602.10029v2
- CALF: https://arxiv.org/abs/2603.12543
- MRLMN: https://arxiv.org/html/2505.08448v2
- POGS-QMIX: https://www.mdpi.com/2504-446X/10/9/688
- Sun 2023: https://www.sciencedirect.com/science/article/abs/pii/S0925231222014862
- Submodular MARL: https://arxiv.org/abs/2603.06810
- COMA: Foerster et al., AAMAS 2018
---
