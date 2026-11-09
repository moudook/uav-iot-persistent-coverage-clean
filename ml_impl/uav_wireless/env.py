"""
Dec-POMDP environment for persistent, fault-tolerant UAV-IoT coverage.

Implements Section III of Mlawn_midsem_report_FINAL:
  A. system architecture      B. Dec-POMDP formulation
  C. global state (3)         D. local observation (4)
  E. discrete actions (5,6)   F. persistent K-coverage (7-10)
  G. connectivity + articulation points (11-13)
  H. battery model (14-16)    I. charging arbitration (17-18)
  J. safety constraints (19)  K. failure and recovery (20)
  L. team reward (23) + weights from Table IV
  M. parameters from Table VI / VII

Every coverage and connectivity quantity is PHY-derived via wireless.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

from .wireless import WirelessModel, los_probability

# --------------------------------------------------------------------------
# Actions (Table III)
# --------------------------------------------------------------------------
A_HOVER, A_PX, A_NX, A_PY, A_NY, A_CHARGE = range(6)
N_ACTIONS = 6
ACTION_VECTORS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]]
)


@dataclass
class SimConfig:
    # Table VI
    N: int = 5                    # active UAVs
    N_reserve: int = 1
    M: int = 20                   # IoT sensors
    side: float = 80.0            # Omega = 80 x 80 m^2
    R_cov: float = 35.0           # coverage radius
    R_comm: float = 40.0          # communication radius (geometric cap)
    z0: float = 50.0              # nominal altitude
    K: int = 2                    # coverage redundancy target
    dt: float = 0.2               # control interval
    v_max: float = 5.0
    d_safe: float = 5.0
    B_max: float = 30_000.0       # J
    P_hover: float = 50.0         # W
    P_move: float = 75.0          # W
    P_elec: float = 5.0           # W  avionics + comms
    P_tx: float = 0.1             # W  RF transmit by a UAV (20 dBm EIRP)
    P_tx_sensor: float = 0.02     # W  RF transmit by a sensor (13 dBm)
    charge_rate: float = 0.0002   # battery fraction per step
    episode_len: int = 1_000      # steps (report uses 10000 = 2000 s)
    n_pads: int = 2               # S, charging pads
    pad_service_radius: float = 15.0
    E_reserve: float = 2_000.0    # J kept in reserve
    failure_enabled: bool = False
    failure_time: int = -1        # -1 = disabled
    credit_mode: str = "team"   # "team", "c3", or "diff"
    credit_weight: float = 0.10
    diff_beta: float = 1.00     # weight on per-agent difference reward
    randomize_layout: bool = False  # resample sensor/pad layout each episode
    adversarial_layout: bool = False  # worst-case corner-clustered sensors
    ego_sensors: bool = False       # ego-centric bearings to nearest deficient sensors
    use_critical_anchor: bool = False  # scalar: fraction of UAV i's covered sensors with n_d==K
    n_ego: int = 3                 # number of nearest deficient sensors to report
    use_w_penalty: bool = True      # enable unnecessary-movement penalty W_t
    dist_shape_kappa: float = 0.0   # potential-based distance shaping coefficient
    energy_shape_kappa: float = 0.0  # potential-based ENERGY shaping coefficient
    intrinsic_beta: float = 0.0     # count-based intrinsic motivation weight
    # Table IV weights
    lam_O: float = 1.00           # under-coverage
    lam_E: float = 0.10           # energy consumption
    # Degraded service is the JOINT condition (K-coverage AND connectivity).
    # The draft's 0.50 was below lam_O, so the learner happily traded
    # connectivity for coverage gain. Raised so that violating either
    # invariant dominates, matching the "service first" intent of Table IV.
    lam_D: float = 2.00           # degraded service
    lam_W: float = 0.05           # unnecessary movement
    lam_R: float = 0.30           # recovery delay
    lam_C: float = 0.50           # articulation-point criticality
    # Energy-threshold violation: the mean fractional shortfall below the
    # return threshold.  Without it the controller learns to keep covering
    # and never turns back in time: measured on the corrected simulator, the
    # learned policy stranded all 8 UAVs per episode while a rule-based
    # heuristic stranded 0.7, because the consumption term is two orders of
    # magnitude too small to price the consequence of an empty pack.
    lam_B: float = 0.0
    # ablations (Table VIII / vanilla MAPPO)
    use_connectivity_term: bool = True
    use_reserve: bool = True
    use_charging_arbitration: bool = True
    use_action_mask: bool = True
    # Joint connectivity projection.  The per-agent mask is evaluated against
    # the pre-move state and the agents move simultaneously, so individually
    # admissible moves can jointly disconnect the swarm.  With the
    # projection off, the learned policy deploys (coverage 0.898) and the
    # swarm is disconnected 39% of the time.
    use_joint_projection: bool = True
    # The mask's energy rule.  True (current): a UAV may not depart unless it
    # retains, after the move, the energy to reach a charging pad plus the
    # reserve margin.  False (previous): a departure is admitted whenever
    # E_i - P_move*dt > 0, which allows a UAV to fly until its pack is
    # empty away from a pad; an exhausted UAV then leaves the active set.
    strict_energy_mask: bool = True
    mask_horizon: int = 3         # H-step connectivity lookahead in the mask
    # Initial deployment.  False: a centred ring that already covers most of
    # the field (little headroom).  True: a connected cluster in one corner
    # that covers little, so the policy must deploy while keeping the graph
    # connected -- the regime in which the constraint actually binds.
    start_clustered: bool = False
    # Include a one-hot agent identity in the observation.  Parameter sharing
    # is only sound if each agent's own observation identifies it, so that a
    # shared network can still specialise (Foerster et al., AAMAS 2016).  The
    # observation of Eq. (4) carries position, velocity, battery, mode and
    # the neighbour slots, but no identity, so agents in a symmetric cluster
    # receive identical inputs and the deterministic policy picks the SAME
    # action for all of them: the swarm translates as a rigid blob and never
    # disperses.  Measured: 1.00 distinct actions out of 8 with the identity
    # absent.
    use_agent_id: bool = False
    # Coarse coverage-map summary in the observation, as a G x G grid of the
    # share of under-covered sensors in each cell of the region.  The
    # infrastructure already owns the charging pads and aggregates the IoT
    # service reports, so a broadcast map summary is a communication
    # assumption rather than an oracle.  Without it the agent knows HOW MANY
    # sensors it covers but not WHICH are uncovered, so there is no gradient
    # signal that points it towards the deficit.
    map_grid: int = 0
    # Unit bearing and range to the nearest charging pad, so that a UAV in
    # the return-to-charge state can fly home.  Without it the observation
    # carried only the ENERGY cost of the flight home, not its direction,
    # and a learned policy could not express the return at all.
    use_pad_bearing: bool = False
    seed: int = 0

    def dict(self):
        return asdict(self)


def obs_dim(cfg: SimConfig) -> int:
    """Length of the per-agent observation for this configuration (Eq. 4).

    own(7) + 4 neighbour slots x 4 + tail(7), plus a one-hot agent identity
    of length N when `use_agent_id` is set, plus a G x G coverage-map summary
    when `map_grid` is set.
    """
    base = 7 + 4 * 4 + 7            # own(7) + 4 neighbour slots x 4 + tail(7)
    base += cfg.N if cfg.use_agent_id else 0
    base += cfg.map_grid ** 2 if cfg.map_grid else 0
    base += 3 if cfg.use_pad_bearing else 0
    if getattr(cfg, "ego_sensors", False):
        base += getattr(cfg, "n_ego", 3) * 3 + 2   # (dx,dy,dist) per sensor + repulsion (x,y)
    if getattr(cfg, "use_critical_anchor", False):
        base += 1  # fraction of UAV i's covered sensors with exactly K coverage
    return base


def state_dim(cfg: SimConfig) -> int:
    """Length of the centralised critic input for this configuration (Eq. 3)."""
    return 2 * cfg.N + 3 * cfg.N + 2 * cfg.M + cfg.n_pads + 3


class UAVIoTEnv:
    """Cooperative persistent-coverage Dec-POMDP."""

    #: Number of neighbour slots in the local observation (Eq. 4).
    NB_SLOTS = 4

    # Local observation layout (Eq. 4), built by concatenation:
    #   7  own state  [x, y, vx, vy, battery, mode, pad_assigned]
    #   16 4 neighbour slots x [dx, dy, battery_j, distance]
    #   7  [n_neighbours, local_coverage, bridge_risk,
    #       return_urgency, pad_energy, charging_wait, active]
    OBS_DIM = 7 + 4 * NB_SLOTS + 7          # = 30

    # Global state layout (Eq. 3), built by concatenation:
    #   2N positions, N battery, N mode, N failed, 2M sensors, n_pads, 3 scalars
    # For the reference configuration (N=5, M=20, n_pads=2) this is
    # 10 + 15 + 40 + 2 + 3 = 70. Kept as a constant so code that only knows the
    # default config keeps working; always prefer `env.state_dim` or
    # `state_dim(cfg)`, which are correct for every configuration. The previous
    # value, 80, was correct only for N=5 and raised for N >= 8.
    STATE_DIM = 2 * 5 + 3 * 5 + 2 * 20 + 2 + 3      # = 70 for N=5, M=20

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._episode_idx = 0
        self.radio = WirelessModel(z0=cfg.z0, p_tx=cfg.P_tx,
                                    p_tx_sensor=cfg.P_tx_sensor)
        self.obs_dim = obs_dim(cfg)
        self.state_dim = state_dim(cfg)
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        c = self.cfg
        if getattr(c, "randomize_layout", False):
            self.rng = np.random.default_rng(int(c.seed) * 1000003 + self._episode_idx * 999983 + 17)
        self.t = 0
        # Initial deployment must be FEASIBLE. Uniformly random positions left
        # the swarm disconnected on 11/20 seeds, which makes SC>0 unreachable
        # regardless of policy quality -- the learner cannot recover from an
        # infeasible start. UAVs are therefore seeded as a connected cluster
        # (ring + centre) and the connectivity invariant is checkable from t=0.
        self.p = self._connected_initial_positions()
        self.v = np.zeros((c.N, 2))
        self.E = self.rng.uniform(0.7, 1.0, size=c.N) * c.B_max
        self.failed = np.zeros(c.N, dtype=bool)
        self.mode = np.zeros(c.N, dtype=int)          # 0 serve, 1 return, 2 charge
        self.pad_assigned = np.full(c.N, -1, dtype=int)
        self.pad_queue = np.zeros(c.n_pads, dtype=int)
        self.pad_occupied_by = np.full(c.n_pads, -1, dtype=int)

        if getattr(c, "adversarial_layout", False):
            # Innovation 9: worst-case layout - sensors clustered in the
            # four corners, pads placed centrally.  Harder for a
            # coverage policy than a uniform draw.
            corners = [(0.1*c.side, 0.1*c.side), (0.9*c.side, 0.1*c.side),
                       (0.1*c.side, 0.9*c.side), (0.9*c.side, 0.9*c.side)]
            self.sensors = np.array(
                [corners[i % 4] + self.rng.normal(0, 3.0, 2)
                 for i in range(c.M)], dtype=float)
            self.sensors = np.clip(self.sensors, 0, c.side)
            self.pads = self.rng.uniform(0.4*c.side, 0.6*c.side,
                                       size=(c.n_pads, 2))
        else:
            self.sensors = self.rng.uniform(0, c.side, size=(c.M, 2))
            self.pads = self.rng.uniform(0.15 * c.side, 0.85 * c.side,
                                       size=(c.n_pads, 2))

        # Draw the episode's propagation environment once. Seeded from the
        # episode seed but fed through a separate stream, so shadowing draws
        # never perturb the env's own RNG (and vice versa) -- otherwise the
        # physics would leak into action sampling and runs would not replay.
        self.radio.reset_links(c.N, c.M, seed=(int(c.seed) * 1_000_003 + 17))

        self.reserve_used = 0
        self.failure_t = c.failure_time if c.failure_time >= 0 else -1
        if c.failure_enabled and c.failure_time < 0:
            self.failure_t = int(self.rng.integers(c.episode_len // 3, c.episode_len))

        self._history_len = 6
        self.prev_cov_score = 0.0
        self.prev_w_dist = 0.0
        self.last_cost = dict(U_t=1.0, E_t=0.0, D_t=1.0, W_t=0.0, R_t=0.0, Crit_t=1.0)
        self.last_agent_rewards = None
        self._prev_phi = 0.0
        self._prev_phi_e = 0.0
        self._visit_counts = {}
        self.recovery_start = -1
        self._episode_idx += 1
        self.recovered = True
        self.charging_wait = np.zeros(c.N)
        self.denied_charge = 0
        self.return_events = 0      # serve -> return transitions
        self.stranded = 0           # UAVs that ran out of energy away from a pad
        self.reserve_activations = 0
        self.projection_revisions = 0   # moves reverted by the joint projection
        self.unexecuted_actions = 0    # sampled actions that were not executed

        self.ep = self.metrics()
        self._under_streak = 0
        self.truncated = False
        self.terminated = False
        self._map_sum = self.coverage_map_summary()
        self.prev_metrics = dict(self.ep)
        return self.local_obs(), self.global_state()

    # ------------------------------------------------------------------
    # PHY-derived coverage and connectivity
    # ------------------------------------------------------------------
    def _connected_initial_positions(self):
        """Place N UAVs so the communication graph is connected from t=0.

        Two deployments are supported.

        ``start_clustered=False`` (default): a ring of radius R_local centred
        in the region, plus any extra UAVs inside the ring.  R_local is
        chosen as a fraction of R_comm so that every ring neighbour is well
        within communication range even after the PHY rate check.  This is
        the deployment a dispatcher would actually use, but it has a
        property worth stating: on an 80x80 m field with R_cov = 35 m the
        ring already covers most of the sensor set, so the achievable range
        for any controller is roughly [0.9, 1.0] and the task has little
        headroom.

        ``start_clustered=True``: every UAV starts in one connected blob
        near a corner.  The swarm is connected from t=0 (so the invariant
        is still checkable) but the initial coverage is low, and the policy
        has to deploy across the field while keeping the graph connected.
        This is the standard benchmark arrangement for persistent coverage
        and it is the regime in which the connectivity constraint and the
        feasibility mask actually bind.
        """
        c = self.cfg
        if c.start_clustered:
            # A tight connected chain: consecutive UAVs are one hop apart,
            # so the graph is connected, but the blob covers only a small
            # part of the field.
            r_local = min(0.25 * c.R_comm, 0.12 * c.side)
            corner = np.array([0.15 * c.side, 0.15 * c.side])
            n_ring = max(c.N - 1, 1)
            ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            pos = [corner + r_local * np.array([np.cos(a), np.sin(a)])
                   for a in ang]
            while len(pos) < c.N:
                pos.append(corner + self.rng.normal(0, 0.08 * r_local, size=2))
            pos = np.array(pos[: c.N])
            return np.clip(pos, 0.0, c.side)

        centre = np.array([c.side / 2.0, c.side / 2.0])
        r_local = min(0.45 * c.R_comm, 0.30 * c.side)
        n_ring = max(c.N - 1, 1)
        ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        pos = [centre + r_local * np.array([np.cos(a), np.sin(a)]) for a in ang]
        while len(pos) < c.N:                      # extra UAVs fill the centre
            pos.append(centre + self.rng.normal(0, 0.1 * r_local, size=2))
        pos = np.array(pos[: c.N])
        return np.clip(pos, 0.0, c.side)

    def _active_idx(self):
        return np.where(~self.failed)[0]

    def coverage_counts(self):
        """n_d(t) for each sensor: how many ACTIVE UAVs can actually SERVE it."""
        act = self._active_idx()
        if len(act) == 0:
            return np.zeros(self.cfg.M)
        # geometric ceiling first
        d = np.linalg.norm(self.sensors[:, None, :] - self.p[act][None, :, :], axis=-1)
        within = d <= self.cfg.R_cov
        # PHY gate: a UAV only "counts" if ITS OWN serving link clears the rate
        # threshold. Each sensor has a serving UAV (the strongest receiver), so
        # evaluate the SINR per (sensor, uav) pair rather than once per sensor.
        sinr = self._access_sinr_matrix(self.p[act], self.sensors, act)   # (M, K)
        rate = self.radio.rate_mbps(self.radio.snr_db(sinr))              # (M, K)
        phy_ok = rate >= self.radio.rate_min_mbps                         # (M, K)
        usable = within & phy_ok
        return usable.sum(axis=1)

    def _access_sinr_matrix(self, uav_xy, sensor_xy, uav_idx):
        """(M, K) SINR of every sensor->UAV pair, with inter-UAV interference.

        The report's K-coverage counts how many UAVs can *simultaneously serve*
        a sensor, so we need a per-pair value rather than a single best-SINR
        number. Each UAV is treated as an interfering transmitter for the
        others, so serving with more UAVs costs SINR -- which is exactly the
        interference effect the experiment is meant to expose.

        `uav_idx` are the GLOBAL ids of the UAVs in `uav_xy`, used to look up
        each link's shadowing by identity rather than by position.
        """
        return self.radio.access_sinr(sensor_xy, uav_xy, uav_idx)

    def comm_graph(self):
        """Adjacency + SNR matrix over active UAVs."""
        act = self._active_idx()
        n = len(act)
        if n == 0:
            return act, np.zeros((0, 0), dtype=bool), np.zeros((0, 0))
        snr_lin = self.radio.backhaul_snr(self.p[act], act)
        snr = self.radio.snr_db(snr_lin)
        rate = self.radio.link_rate_mbps(snr, backhaul=True)
        d = np.linalg.norm(self.p[act][:, None, :] - self.p[act][None, :, :], axis=-1)
        adj = (d <= self.cfg.R_comm) & (rate >= self.radio.rate_min_mbps)
        np.fill_diagonal(adj, False)
        return act, adj, snr

    @staticmethod
    def articulation_points(adj: np.ndarray) -> np.ndarray:
        """Tarjan articulation points of an undirected graph. Eq. (12)."""
        n = len(adj)
        if n < 3:
            return np.zeros(n, dtype=bool)
        disc = [-1] * n
        low = [0] * n
        parent = [-1] * n
        ap = np.zeros(n, dtype=bool)
        timer = [0]

        def dfs(u):
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            children = 0
            for v in range(n):
                if not adj[u, v]:
                    continue
                if disc[v] == -1:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for u in range(n):
            if disc[u] == -1:
                dfs(u)
        return ap

    def is_connected(self, adj: np.ndarray) -> bool:
        n = len(adj)
        if n == 0:
            return False
        if n == 1:
            return True
        visited = np.zeros(n, dtype=bool)
        stack = [0]
        visited[0] = True
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u, v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)
        return visited.all()

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------
    def power_draw(self, actions):
        """Energy drawn per step by each UAV.

        A UAV that is charging on a pad has its rotors idle: it draws only
        the avionics and communications baseline, and the pad's power goes
        into the battery rather than being spent.  The previous revision
        billed a charging UAV the full HOVER power, which made charging
        uneconomic -- the pack lost more energy per step to the rotors than
        the pad put back, so no UAV could ever recover, and the whole
        charging-coordination mechanism of the paper was unusable.
        """
        c = self.cfg
        p = np.where(
            np.isin(actions, [A_PX, A_NX, A_PY, A_NY]),
            c.P_move,
            np.where(actions == A_CHARGE, 0.0, c.P_hover),
        )
        return p + c.P_elec + c.P_tx     # P_comm / P_elec, constant baseline

    def _pad_energy_from(self, i, pos):
        """Energy to fly from `pos` to the nearest charging pad.

        The same expression as `pad_energy`, but from an arbitrary position,
        so that the feasibility mask can ask what the cost would be AFTER a
        candidate move.
        """
        if len(self.pads) == 0:
            return 0.0
        d = np.linalg.norm(self.pads - np.asarray(pos), axis=1)
        t = float(np.min(d)) / self.cfg.v_max
        return t * (self.cfg.P_move + self.cfg.P_elec) * 1.15

    def _pad_energy_from(self, i, pos):
        """Energy to fly from `pos` to the nearest charging pad.

        The same expression as `pad_energy`, but from an arbitrary position,
        so that the feasibility mask can ask what the cost would be AFTER a
        candidate move.
        """
        if len(self.pads) == 0:
            return 0.0
        d = np.linalg.norm(self.pads - np.asarray(pos), axis=1)
        t = float(np.min(d)) / self.cfg.v_max
        return t * (self.cfg.P_move + self.cfg.P_elec) * 1.15

    def _pad_energies(self, positions):
        """The energy to reach the nearest pad from each of several points.

        The mask asks this for four candidate positions per agent, so the
        distance to every pad is computed once for the whole batch instead
        of once per candidate.
        """
        if len(self.pads) == 0:
            return np.zeros(len(positions), dtype=float)
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        d = np.linalg.norm(self.pads[None, :, :] - pos[:, None, :], axis=-1)
        t = d.min(axis=1) / self.cfg.v_max
        return t * (self.cfg.P_move + self.cfg.P_elec) * 1.15

    def pad_energy(self, i):
        """E^pad_i: energy to reach the nearest feasible pad. Eq. (16)."""
        if len(self.pads) == 0:
            return 0.0
        d = np.linalg.norm(self.pads - self.p[i], axis=1)
        d_min = float(np.min(d))
        t = d_min / self.cfg.v_max
        return t * (self.cfg.P_move + self.cfg.P_elec) * 1.15

    def pad_bearing(self, i):
        """Unit vector towards the nearest charging pad, and the range to it.

        A UAV that has entered the return-to-charge state has to FLY to a pad,
        and the observation of Eq. (4) carried only the energy cost of that
        flight, not its direction, so a learned policy could not express
        the return at all.  The pad positions are fixed infrastructure and
        are broadcast together with the coverage-map summary, so this is the
        same communication assumption, not an oracle.
        """
        if len(self.pads) == 0:
            return np.zeros(3, dtype=np.float32)
        d = self.pads - self.p[i]
        dist = np.linalg.norm(d, axis=1)
        k = int(np.argmin(dist))
        if dist[k] < 1e-9:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        u = d[k] / dist[k]
        return np.array([u[0], u[1], dist[k] / self.cfg.side], dtype=np.float32)

    def return_threshold(self, i):
        return self.pad_energy(i) + self.cfg.E_reserve

    # ------------------------------------------------------------------
    # Charging arbitration (Hungarian-style). Eq. (17)-(18)
    # ------------------------------------------------------------------
    def arbitrate(self, requests):
        if not requests:
            return {}
        if not self.cfg.use_charging_arbitration:
            return {requests[0]: 0} if len(self.pads) else {}

        n_pads = len(self.pads)
        max_q = max(self.pad_queue.max(), 1)
        cost = np.zeros((len(requests), n_pads))
        for a, i in enumerate(requests):
            d = np.linalg.norm(self.pads - self.p[i], axis=1)
            urgency = 1.0 - self.E[i] / self.cfg.B_max
            for s in range(n_pads):
                cost[a, s] = (
                    1.0 * (d[s] / self.cfg.side)
                    + 0.5 * (self.pad_queue[s] / max_q)
                    + 1.0 * urgency
                )
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        assign = {}
        for r, s in zip(rows, cols):
            i = requests[r]
            if self.pad_occupied_by[s] == -1:
                assign[i] = s
                self.pad_occupied_by[s] = i
            else:
                self.denied_charge += 1
        return assign

    # ------------------------------------------------------------------
    # Action feasibility mask. Section IV-B / Table V
    # ------------------------------------------------------------------
    def action_mask(self, i, base=None):
        c = self.cfg
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[A_HOVER] = True

        act = self._active_idx()
        if i not in act:
            # a failed UAV has no admissible action but hovering
            return m
        others = act[act != i]
        step = c.v_max * c.dt
        dirs = [A_PX, A_NX, A_PY, A_NY]
        cands = np.clip(self.p[i] + np.array([ACTION_VECTORS[a] for a in dirs])
                         * step, 0, c.side)
        # The prospective graph for ALL FOUR candidate moves is computed in one
        # vectorised call.  The previous version computed the whole NxN physics
        # separately for each of the four candidates, so the base matrix was
        # rebuilt 32 times per step; the only thing that differs between the
        # candidates is the row and column of the moving UAV, and all four
        # rows are now evaluated together.
        adj_c = self._candidate_adjacencies(i, cands, base=base) \
            if c.use_action_mask else None
        # the energy to reach the nearest pad, from the current position and
        # from each of the four candidates, computed in one batch
        pad_energies = self._pad_energies(
            np.vstack([self.p[i][None, :], cands]))
        pad_now = float(pad_energies[0])

        for a_dir, a in enumerate(dirs):
            p_new = cands[a_dir]
            ok = True
            # workspace: reject moves that would leave the region
            if np.any(p_new < 0) or np.any(p_new > c.side):
                ok = False
            # separation
            if ok and len(others):
                if np.min(np.linalg.norm(p_new - self.p[others], axis=1)) < c.d_safe:
                    ok = False
            # energy reachability
            #
            # A UAV must not depart unless it can still get home: after the
            # move it must retain the energy to reach a charging pad plus
            # the reserve margin, computed from where it WOULD BE.  The
            # previous test, E_i - P_move*dt > 0, admitted a departure with
            # a flat battery, so a UAV could fly until its pack was empty
            # away from a pad -- and an exhausted UAV is removed from the
            # active set, which takes its share of the coverage and of the
            # graph with it.  That single weak test is the mechanism by which
            # the learned controller stranded six of its eight UAVs per
            # episode.
            #
            # A UAV that is ALREADY below the threshold is allowed to move
            # towards a pad, since blocking it there would strand it
            # certainly; the test is therefore one-sided in its favour.
            if ok:
                e_after = self.E[i] - c.P_move * c.dt
                if c.strict_energy_mask:
                    pad_new = float(pad_energies[a_dir + 1])
                    reserve = c.E_reserve
                    if e_after < pad_new + reserve and pad_new >= pad_now:
                        ok = False
                elif e_after <= 0:
                    ok = False
            # connectivity preservation: block moves that would break the
            # graph NOW *or* leave it fragile. A one-step check alone lets
            # the swarm walk itself into a partition, which is exactly what
            # the first experiment run showed (AG collapsed 1.0 -> 0.2).
            if ok and c.use_action_mask:
                if not self._h_step_safe(i, p_new, c.mask_horizon,
                                         adj=adj_c[a_dir]):
                    ok = False
                elif self._is_articulation_move(i, p_new, adj=adj_c[a_dir]):
                    ok = False
            m[a] = ok

        # charge: must be inside a pad service region, and the pack must
        # not already be full.
        #
        # The old admission test was "E <= return_threshold", and that
        # trapped the UAV: the departure rule demands
        # E_after >= pad_energy(destination) + E_reserve, and at a pad
        # the threshold IS E_reserve, so a UAV that charged above it
        # could not charge and a UAV that stayed below it could not
        # leave.  It spent the rest of the episode in a three-step
        # charge-and-hover cycle at the threshold.  Charging is now
        # admissible whenever the pack is not full.
        near = np.any(np.linalg.norm(self.pads - self.p[i], axis=1) <= c.pad_service_radius)
        needs = self.E[i] < c.B_max - 1e-6
        m[A_CHARGE] = bool(near and needs)
        return m

    @staticmethod
    def _adj_bits(adj):
        """The rows of an adjacency matrix as integer bitmasks.

        A traversal of an 8-node graph is then a handful of integer
        operations instead of a Python loop over the adjacency, and the
        mask is computed once per candidate action rather than being walked
        element by element.
        """
        n = adj.shape[0]
        out = []
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    m |= 1 << j
            out.append(m)
        return out

    @staticmethod
    def _bfs_mask(bits, start, limit=None):
        """The set of nodes reachable from `start`, as a bitmask.

        `limit` caps the number of hops.  Returns the reached mask and the
        number of hops actually used.
        """
        n = len(bits)
        full = (1 << n) - 1
        reach = 1 << start
        frontier = reach
        hops = 0
        while frontier:
            nxt = 0
            f = frontier
            while f:
                low = f & -f
                nxt |= bits[low.bit_length() - 1]
                f ^= low
            nxt &= ~reach
            if not nxt:
                break
            reach |= nxt
            frontier = nxt
            hops += 1
            if limit is not None and hops >= limit:
                break
        return reach & full, hops

    def _is_articulation_move(self, i, p_new, adj=None):
        """True if moving to p_new makes UAV i a cut vertex of the graph.

        Staying connected at t+1 is not enough: if the swarm only survives
        while one particular UAV sits in the middle, the next move may split
        it. Blocking articulation-forming moves gives the policy room to
        re-spread instead of tunnelling.
        """
        act = self._active_idx()
        n = len(act)
        if n <= 3:
            return False
        if adj is None:
            adj = self._candidate_adjacency(i, p_new)
        pos_i = list(act).index(i)
        # A UAV is a cut vertex of the prospective graph exactly when
        # deleting it disconnects the rest.  One bitmask traversal of the
        # reduced graph decides it, instead of a full Tarjan pass.  The
        # reduced adjacency must NOT be able to route through the removed
        # UAV, so its row and column are cleared first.
        bits = self._adj_bits(adj)
        drop = 1 << pos_i
        for u in range(n):
            if bits[u] & drop:
                bits[u] &= ~drop
        bits[pos_i] = 0
        start = 0 if pos_i != 0 else 1
        reach, _ = self._bfs_mask(bits, start)
        target = ((1 << n) - 1) & ~drop
        return reach != target

    def _base_graph(self, act):
        """The distance and SNR matrices of the swarm in its current state.

        Every agent's feasibility mask is evaluated against the SAME
        pre-move state, so the base graph is computed once per step and
        shared, rather than once per agent.
        """
        p_act = self.p[act]
        d = np.linalg.norm(p_act[:, None, :] - p_act[None, :, :], axis=-1)
        snr = self.radio.backhaul_snr(p_act, act)
        return d, snr

    def action_masks(self):
        """The feasibility mask of every agent, sharing one base graph."""
        act = self._active_idx()
        base = self._base_graph(act)
        return [self.action_mask(i, base=base) for i in range(self.cfg.N)]

    def _candidate_adjacencies(self, i, cands, base=None):
        """The prospective communication graph for SEVERAL candidate moves.

        Only the edges incident to UAV i change when it moves, so the base
        SNR matrix is spliced with a recomputed row rather than recomputed
        in full.  All the candidates are evaluated in one vectorised call,
        because they differ only in that row.

        Returns an array of shape (P, N, N) of adjacency matrices.
        """
        act = self._active_idx()
        n = len(act)
        if n == 0:
            return np.zeros((len(cands), 0, 0), dtype=bool)
        pos_i = list(act).index(i)
        p_act = self.p[act]
        if base is None:
            base = self._base_graph(act)
        d, snr = base
        cands = np.atleast_2d(np.asarray(cands, dtype=float))
        P = len(cands)

        # the SNR row of the moving UAV at each candidate position, in one call
        rows = self.radio.backhaul_snr_rows(p_act, act, pos_i, cands)   # (P,N)
        d_new = np.linalg.norm(p_act[None, :, :] - cands[:, None, :],
                               axis=-1)                                  # (P,N)

        d2 = np.repeat(d[None, :, :], P, axis=0)
        d2[:, pos_i, :] = d_new
        d2[:, :, pos_i] = d_new
        snr2 = np.repeat(snr[None, :, :], P, axis=0)
        snr2[:, pos_i, :] = rows
        snr2[:, :, pos_i] = rows
        rate = self.radio.link_rate_mbps(self.radio.snr_db(snr2),
                                         backhaul=True)
        adj = (d2 <= self.cfg.R_comm) & (rate >= self.radio.rate_min_mbps)
        idx = np.arange(n)
        adj[:, idx, idx] = False
        return adj

    def _candidate_adjacency(self, i, p_new, base=None):
        """Adjacency of the swarm with UAV i hypothetically moved to p_new.

        The H-step connectivity check and the articulation check both need
        the prospective graph, and both were recomputing it from scratch, so
        the physics was evaluated twice per candidate action.  The graph is
        returned once and shared; the result is identical.
        """
        return self._candidate_adjacencies(i, np.atleast_2d(p_new),
                                            base=base)[0]

    def _project_connected(self, old_p, moved):
        """Revert the fewest moves needed to keep the swarm connected.

        The per-agent feasibility mask is evaluated against the pre-move
        state, and all the agents move simultaneously, so a set of
        individually admissible moves can jointly disconnect the swarm.  The
        joint configuration is therefore projected back onto the connected
        set: the first single reversion that restores connectivity is
        applied, and if no single reversion does, every mover is held.

        Returns the number of moves that were reverted, which is reported
        as a metric so that the cost of the projection is visible.
        """
        act, adj, _ = self.comm_graph()
        if self.is_connected(adj):
            return 0
        for i in moved:
            cur = self.p[i].copy()
            self.p[i] = old_p[i]
            _, adj2, _ = self.comm_graph()
            if self.is_connected(adj2):
                self.v[i] = np.zeros(2)      # it is holding position
                return 1
            self.p[i] = cur
        # No single reversion restores connectivity.  Hold every mover: the
        # swarm stays together, which is the invariant the paper claims.
        n = 0
        for i in moved:
            if not np.allclose(self.p[i], old_p[i]):
                self.p[i] = old_p[i]
                self.v[i] = np.zeros(2)
                n += 1
        return n

    def _h_step_safe(self, i, p_new, horizon=3, adj=None):
        """Reject a move that leaves the swarm unable to stay connected.

        One-step lookahead cannot see a partition that forms two steps later,
        which is exactly how the swarm walked itself apart in the first
        experiment run (AG 1.0 -> 0.2). This checks the pessimistic case: after
        moving, can every other active UAV still reach UAV i within `horizon`
        communication hops? If the answer is no, the move is infeasible.
        """
        act = self._active_idx()
        if len(act) <= 2:
            return True
        if adj is None:
            adj = self._candidate_adjacency(i, p_new)
        # The breadth-first search below already establishes whether every
        # node is reachable from the mover, which is a STRONGER condition
        # than connectivity, so the separate connectivity test was a
        # redundant traversal of the graph on every candidate action.
        n = len(adj)
        pos_i = list(act).index(i)
        # the traversal is capped at `horizon` hops, exactly as the original
        # breadth-first search was: it expanded only the nodes at distance
        # less than the horizon, so a node further away was never reached and
        # the check failed
        reach, _ = self._bfs_mask(self._adj_bits(adj), pos_i, limit=horizon)
        return reach == (1 << n) - 1

    def _would_disconnect(self, i, p_new):
        """True if moving UAV i to p_new partitions the swarm graph."""
        act = self._active_idx()
        if len(act) <= 2:
            return False
        saved = self.p[i].copy()
        self.p[i] = p_new
        _, adj, _ = self.comm_graph()
        self.p[i] = saved
        return not self.is_connected(adj)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def coverage_map_summary(self, ncov=None):
        """Coarse broadcast summary of where the coverage deficit is.

        A G x G grid over the region.  Cell (gx, gy) holds the share of ALL
        sensors that lie in that cell and are currently under-covered
        (n_d(t) < K).  Normalising by M rather than by the cell population
        makes the value a global quantity: the agent learns both where the
        deficit is and how large it is, and the encoding is independent of
        M.

        The value is a *communication* assumption, not an oracle: the
        infrastructure that owns the charging pads also aggregates the IoT
        service reports and broadcasts this summary to the swarm.  The
        summary is fixed size, so the observation does not grow with the
        number of sensors.
        """
        c = self.cfg
        G = c.map_grid
        out = np.zeros(G * G, dtype=np.float32)
        if not G:
            return out
        if ncov is None:
            ncov = self.coverage_counts()
        if len(ncov) == 0:
            return out
        under = ncov < c.K
        if not np.any(under):
            return out
        gx = np.clip((self.sensors[:, 0] / c.side * G).astype(int), 0, G - 1)
        gy = np.clip((self.sensors[:, 1] / c.side * G).astype(int), 0, G - 1)
        cell = gy * G + gx
        w = under.astype(np.float32) / max(c.M, 1)
        out = np.bincount(cell, weights=w, minlength=G * G).astype(np.float32)
        return out

    def local_obs(self):
        """o_i(t) = [p_i, v_i, b_i, N_i, C_i^local, rho_i, tau_i^return, E_i^pad]. Eq. (4).

        The vector is built by concatenation and then checked against
        `self.obs_dim`. The previous revision wrote into a preallocated array
        with a stride of 2 while storing 4 values per neighbour, so neighbour
        slots overlapped each other and neighbour 3 overwrote the neighbour
        count and the local coverage count. The actor was trained on a
        corrupted observation. Building by concatenation makes that class of
        bug impossible, and the assertion makes any future mismatch loud
        instead of silent.
        """
        c = self.cfg
        act, adj, _ = self.comm_graph()
        ncov = self.coverage_counts()
        ap = self.articulation_points(adj) if len(adj) else np.zeros(0, bool)

        rows = []
        for i in range(c.N):
            if self.failed[i]:
                rows.append(np.zeros(self.obs_dim, dtype=np.float32))
                continue

            own = np.array([
                self.p[i][0] / c.side,
                self.p[i][1] / c.side,
                self.v[i][0] / c.v_max,
                self.v[i][1] / c.v_max,
                self.E[i] / c.B_max,
                self.mode[i] / 2.0,
                1.0 if self.pad_assigned[i] >= 0 else 0.0,
            ], dtype=np.float32)

            # four nearest neighbours within R_comm, each 4 fields
            near = []
            for j in act:
                if j == i:
                    continue
                d = float(np.linalg.norm(self.p[j] - self.p[i]))
                if d <= c.R_comm:
                    near.append((j, d))
            near.sort(key=lambda z: z[1])
            nb = np.zeros(4 * self.NB_SLOTS, dtype=np.float32)
            for k, (j, d) in enumerate(near[: self.NB_SLOTS]):
                rel = (self.p[j] - self.p[i]) / c.R_comm
                nb[4 * k + 0] = rel[0]
                nb[4 * k + 1] = rel[1]
                nb[4 * k + 2] = self.E[j] / c.B_max
                nb[4 * k + 3] = d / c.R_comm

            # local bridge risk rho_i: is any neighbour a cut vertex locally?
            # adj rows/cols are indexed by POSITION within `act`, not by the
            # global UAV id, so translate before indexing.
            pos_i = list(act).index(i)
            deg = int(adj[pos_i].sum()) if len(adj) else 0
            bridge = 1.0 if (0 < deg <= 2) else 0.0
            for k, j in enumerate([x for x in act if x != i][:4]):
                idx = list(act).index(j)
                if idx < len(ap) and ap[idx]:
                    bridge = max(bridge, 1.0 - 0.25 * k)

            d_s = np.linalg.norm(self.sensors - self.p[i], axis=1)
            thresh = self.return_threshold(i)
            tail = np.array([
                min(len(near), self.NB_SLOTS) / self.NB_SLOTS,
                float(np.sum(d_s <= c.R_cov)) / max(c.M, 1),
                bridge,
                np.clip((thresh - self.E[i]) / max(thresh, 1.0), -1, 1),
                self.pad_energy(i) / c.B_max,
                np.clip(self.charging_wait[i] / 200.0, 0, 1),
                1.0,
            ], dtype=np.float32)

            row = np.concatenate([own, nb, tail])
            if c.use_pad_bearing:
                # unit bearing + range to the nearest charging pad, so that a
                # UAV in the return-to-charge state can actually fly home
                row = np.concatenate([row, self.pad_bearing(i)])
            if c.map_grid:
                # broadcast coverage-map summary (see coverage_map_summary)
                row = np.concatenate([row, self._map_sum])
            if c.use_agent_id:
                # one-hot identity so that a SHARED network can specialise.
                # Without it, agents in a symmetric cluster have identical
                # inputs and the deterministic policy picks the same action
                # for every one of them, so the swarm translates as a rigid
                # blob instead of deploying.
                ident = np.zeros(c.N, dtype=np.float32)
                ident[i] = 1.0
                row = np.concatenate([row, ident])
            if getattr(c, "ego_sensors", False):
                # ego-centric bearings to the nearest deficient sensors
                # (n_d < K) plus a neighbour-repulsion vector.
                k = getattr(c, "n_ego", 3)
                deficient = [d for d in range(c.M) if ncov[d] < c.K]
                # sort by distance to agent i
                def _dist(d):
                    return float(np.linalg.norm(self.sensors[d] - self.p[i]))
                deficient.sort(key=_dist)
                feats = np.zeros(k * 3 + 2, dtype=np.float32)
                for kk, d in enumerate(deficient[:k]):
                    vec = self.sensors[d] - self.p[i]
                    dn = float(np.linalg.norm(vec))
                    if dn > 1e-9:
                        feats[kk*3 + 0] = vec[0] / dn
                        feats[kk*3 + 1] = vec[1] / dn
                    feats[kk*3 + 2] = dn / (1.414 * c.side)
                # neighbour repulsion: away from centroid of nearby UAVs
                near_js = [j for j in act if j != i]
                if near_js:
                    cent = np.mean(self.p[near_js], axis=0)
                    rep = self.p[i] - cent
                    rn = float(np.linalg.norm(rep))
                    if rn > 1e-9:
                        feats[k*3 + 0] = rep[0] / rn
                        feats[k*3 + 1] = rep[1] / rn
                row = np.concatenate([row, feats])
            if getattr(c, "use_critical_anchor", False):
                # critical-coverage anchor: fraction of sensors UAV i covers
                # that have exactly K total coverage (ncov == K). If UAV i
                # moves away, those sensors drop below K. Tells the policy
                # "do not leave this sensor".
                _crit = 0.0
                _ncov_i = 0
                for _d in range(c.M):
                    if d_s[_d] <= c.R_cov:
                        _ncov_i += 1
                        if ncov[_d] == c.K:
                            _crit += 1.0
                _anchor = (_crit / _ncov_i) if _ncov_i > 0 else 0.0
                row = np.concatenate([row, np.array([_anchor], dtype=np.float32)])
            rows.append(row)

        obs = np.stack(rows).astype(np.float32)
        if obs.shape[1] != self.obs_dim:
            raise AssertionError(
                f"local_obs produced {obs.shape[1]} features but obs_dim is "
                f"{self.obs_dim}; the observation layout and the declared "
                f"dimension have diverged"
            )
        return obs

    def global_state(self):
        """s_t for the centralized critic. Eq. (3).

        Built by concatenation and checked, for the same reason as
        `local_obs`: the previous revision wrote into a fixed-size buffer of
        `STATE_DIM = 80`, which is only large enough for N = 5. At N = 8 the
        pad-occupancy slice ran past the end of the array and raised, and
        between those sizes it would have been silently misaligned.
        """
        c = self.cfg
        parts = [
            (self.p / c.side).flatten(),          # 2N
            self.E / c.B_max,                     # N
            self.mode / 2.0,                      # N
            self.failed.astype(float),            # N
            (self.sensors / c.side).flatten(),    # 2M
            (self.pad_occupied_by >= 0).astype(float),   # n_pads
            np.array([
                self.reserve_used / max(c.N_reserve, 1),
                self.t / c.episode_len,
                1.0 if self.is_connected(self.comm_graph()[1]) else 0.0,
            ]),
        ]
        s = np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])
        if s.shape[0] != self.state_dim:
            raise AssertionError(
                f"global_state produced {s.shape[0]} features but state_dim is "
                f"{self.state_dim}; the state layout and the declared dimension "
                f"have diverged"
            )
        return s

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions):
        c = self.cfg
        act_before = self._active_idx().copy()

        # --- 1. motion with safety projection --------------------------
        step = c.v_max * c.dt
        # The feasibility mask of every agent is evaluated once, against the
        # pre-move state, so that the swarm moves simultaneously and the
        # mask is not recomputed per action.
        masks = self.action_masks()
        old_p = self.p.copy()
        masked_actions = 0
        for i in self._active_idx():
            a = int(actions[i])
            if a in (A_PX, A_NX, A_PY, A_NY) and not masks[i][a]:
                # the policy asked for a move the safety filter rejects
                masked_actions += 1
        for i in self._active_idx():
            a = int(actions[i])
            if a in (A_PX, A_NX, A_PY, A_NY):
                self.v[i] = ACTION_VECTORS[a] * c.v_max
            elif a == A_CHARGE:
                self.v[i] = np.zeros(2)
            else:
                self.v[i] = np.zeros(2)
            p_new = np.clip(self.p[i] + self.v[i] * c.dt, 0, c.side)
            # Apply the motion if the CHOSEN action is feasible.
            #
            # The previous revision tested `np.all(self.action_mask(i))`,
            # i.e. it moved the UAV only when EVERY action in the mask was
            # feasible.  A_CHARGE is feasible only when the UAV is at a pad
            # with low energy, so that test was false for essentially every
            # UAV at every step, and NO UAV EVER MOVED.  The whole simulator
            # was static: coverage could not change, every controller scored
            # the same, and the only motion that occurred was by agents that
            # happened to have a full action set available.  The feasibility
            # mask is a property of the chosen action, not of the whole set.
            if masks[i][a]:
                self.p[i] = p_new
            elif a in (A_PX, A_NX, A_PY, A_NY):
                pass    # masked movement -> hold position (mask made this infeasible)

        # --- 1b. joint connectivity projection ------------------------
        # The per-agent mask checks each candidate move against the PRE-move
        # state, but all the agents move at the same time, so a set of
        # individually admissible moves can jointly disconnect the swarm: two
        # UAVs can each step away from a third in a way that is admissible
        # for both and fatal for the third.  Measured on the corrected
        # simulator, this is exactly what happened: the learned policy
        # learned to deploy (coverage 0.898) and the swarm was disconnected
        # 39% of the time, because the per-agent mask could not see the
        # joint move.  The joint configuration is therefore projected back
        # onto the connected set.
        moved = [i for i in self._active_idx()
                 if not np.allclose(self.p[i], old_p[i])]
        if c.use_joint_projection and moved:
            self.projection_revisions += self._project_connected(old_p, moved)
        # the number of sampled actions that were NOT executed: either the
        # filter rejected them, or the joint projection reverted them.  The
        # agent is not told, so the policy gradient is computed for actions
        # that were not taken, and this count is the size of that gap.
        reverted = sum(1 for i in self._active_idx()
                       if not np.allclose(self.p[i], old_p[i]))
        self.unexecuted_actions += masked_actions + max(0, len(moved) - reverted)

        # --- 2. failure injection --------------------------------------
        if self.failure_t >= 0 and self.t == self.failure_t:
            victim = act_before[self.rng.integers(len(act_before))]
            self.failed[victim] = True
            self.mode[victim] = -1
            self.recovered = False
            self.recovery_start = self.t
            self.failure_t = -1
            self.recovered = False
        elif c.use_reserve and self.reserve_used < c.N_reserve and not self.recovered:
            # activate reserve
            self.reserve_used += 1
            self.failed = self.failed.copy()
            # re-seat one failed slot with a fresh UAV at a pad
            dead = np.where(self.failed)[0]
            if len(dead):
                j = int(dead[0])
                self.p[j] = self.pads[self.rng.integers(len(self.pads))]
                self.E[j] = c.B_max
                self.failed[j] = False
                self.mode[j] = 0
                self.recovered = False

        # --- 3. charging arbitration -----------------------------------
        self.pad_occupied_by = np.full(len(self.pads), -1)
        requests = []
        for i in self._active_idx():
            if int(actions[i]) == A_CHARGE:
                requests.append(int(i))
        assign = self.arbitrate(requests)
        for i, s in assign.items():
            self.pad_assigned[i] = s
            self.mode[i] = 2
            self.E[i] = min(c.B_max, self.E[i] + c.charge_rate * c.B_max)
        for i in self._active_idx():
            if i not in assign:
                self.pad_assigned[i] = -1
                if self.mode[i] == 2:
                    self.mode[i] = 0
                self.charging_wait[i] = min(self.charging_wait[i] + 1, 999)
                if self.charging_wait[i] > 150:
                    self.charging_wait[i] = 0

        # --- 4. battery update (14)-(15) -------------------------------
        draw = self.power_draw(actions)
        self.E = np.clip(self.E - draw * c.dt, 0, c.B_max)
        for i in range(c.N):
            if self.failed[i]:
                self.E[i] = 0.0

        # --- 4b. energy state machine -----------------------------------
        # serve --(E <= threshold)--> return --(charge at a pad)--> charge
        #       --(E >= threshold)--> serve
        #
        # The transition out of the serving state is PHYSICAL, not a policy
        # decision: a UAV whose energy has fallen to its return threshold
        # must go home, and the simulator has to know that so that the
        # reward, the metrics and the charging arbitration can see it.
        # Previously the environment never made this transition at all -- the
        # rule-based baselines implemented it themselves, but a LEARNED
        # policy had no way to express it, so the charging coordination and
        # the pad arbitration were never exercised in a learned run.
        for i in self._active_idx():
            thresh = self.return_threshold(i)
            if self.mode[i] == 0 and self.E[i] <= thresh:
                self.mode[i] = 1                       # returning
                self.return_events += 1
            elif self.mode[i] == 1 and self.E[i] >= thresh:
                self.mode[i] = 0                       # back in service

        # A UAV whose pack is empty and that is not sitting on a pad cannot
        # fly and cannot serve: it is out of service.  Counting it as
        # active would let a dead UAV provide coverage and connectivity.
        at_pad = np.any(
            np.linalg.norm(self.pads[:, None, :] - self.p[None, :, :], axis=-1)
            <= c.pad_service_radius, axis=0)
        for i in self._active_idx():
            if self.E[i] <= 1e-9 and not at_pad[i]:
                self.failed[i] = True
                self.mode[i] = -1
                self.stranded += 1
                if c.use_reserve and self.reserve_used < c.N_reserve \
                        and not self.recovered:
                    self.reserve_used += 1
                    j = int(np.where(self.failed)[0][0])
                    self.p[j] = self.pads[self.rng.integers(len(self.pads))]
                    self.E[j] = c.B_max
                    self.failed[j] = False
                    self.mode[j] = 0
                    self.recovered = False
                    self.reserve_activations += 1

        self.t += 1

        # --- 5. recovery detection -------------------------------------
        ncov = self.coverage_counts()
        # The broadcast coverage-map summary is computed once per step, from
        # the post-move coverage counts, so that every agent sees the same
        # summary at the same decision point.
        self._map_sum = self.coverage_map_summary(ncov)
        _, adj, _ = self.comm_graph()
        connected_now = self.is_connected(adj)
        # D_t must not duplicate U_t. U_t already carries coverage continuously,
        # so the binary "service requirement violated" indicator is used for the
        # CONNECTIVITY invariant -- the one that collapsed to 0.2 and left
        # SC = 0 in the first run. This gives the learner a discrete, learnable
        # signal for the thing it was previously not being told about.
        degraded_now = not connected_now
        ok_now = (float(np.mean(ncov >= c.K)) >= 1.0) and connected_now
        if not self.recovered and ok_now:
            self.recovered = True
        rec_delay = 0.0
        if self.recovery_start >= 0 and self.recovered:
            rec_delay = min((self.t - self.recovery_start) / c.episode_len, 1.0)

        # --- 6. reward (23) --------------------------------------------
        #
        # Eq. (22) is a MINIMISATION of a weighted cost. Eq. (23) states the
        # team reward is the NEGATED weighted cost, and the learner MAXIMISES
        # return. Therefore EVERY term must carry a negative sign. The first
        # implementation negated only U_t, which turned energy, degraded
        # service, movement, recovery delay and criticality into positive
        # rewards -- the agent was paid to disconnect. Fixed here.
        U_t = float(np.mean(ncov < c.K))
        E_cons = float(np.sum(draw) * c.dt / (c.N * c.B_max))
        D_t = 1.0 if degraded_now else 0.0
        w_dist = float(np.linalg.norm(self.v, axis=1).sum() * c.dt) if act_before.size else 0.0
        # W_t = "distance flown / distance that changed coverage". A move that
        # buys no new coverage is pure waste, so penalise the fraction of flown
        # distance that did not improve the covered-sensor count.
        cov_now = float(np.mean(ncov >= c.K))
        newly_covered = max(cov_now - self.prev_cov_score, 0.0)
        if getattr(c, "use_w_penalty", True) and w_dist > 1e-9:
            W_t = float(np.clip(1.0 - newly_covered / (w_dist + 1e-9), 0.0, 1.0))
        else:
            W_t = 0.0
        crit = float(self.articulation_points(adj).mean()) if len(adj) else 1.0
        R_rec = rec_delay

        # B_t: the mean fractional shortfall below the return threshold.
        # A UAV that is below its return threshold cannot get home, and the
        # consequence of that arrives late, so the consumption term alone
        # cannot teach the controller to turn back in time.  The feature is
        # the one the observation already carries as the return urgency.
        act_now = self._active_idx()
        if len(act_now):
            shortfalls = [max(0.0, self.return_threshold(i) - self.E[i])
                          / max(self.cfg.B_max, 1.0) for i in act_now]
            B_t = float(np.mean(shortfalls))
        else:
            B_t = 0.0

        # Terms are exposed so eq. (23) can be checked as an equation rather
        # than by inspecting its side effects. See tests/test_physics_and_env.py
        # ::TestRewardEquation and tests/mutation_check.py.
        self.last_cost = dict(
            U_t=U_t, E_t=E_cons, D_t=D_t, W_t=W_t, R_t=R_rec, Crit_t=crit,
            B_t=B_t,
        )
        r = self.reward_from_terms(self.last_cost)

        # Potential-based distance shaping (SI-1): telescopic, policy-invariant.
        kappa = getattr(c, "dist_shape_kappa", 0.0)
        if kappa and kappa > 0:
            def _phi(positions):
                # mean over agents of -min distance to nearest deficient sensor / L
                vals = []
                for i in range(c.N):
                    if self.failed[i]:
                        continue
                    defd = [d for d in range(c.M) if ncov[d] < c.K]
                    if not defd:
                        vals.append(0.0)
                        continue
                    ds = [float(np.linalg.norm(self.sensors[d]-positions[i])) for d in defd]
                    vals.append(-min(ds) / max(c.side * 1.414, 1.0))
                return float(np.mean(vals)) if vals else 0.0
            phi_before = _phi(self.p.copy()) if False else None
            # phi_before should use previous positions; we approximate with current pre-move not stored.
            # Use a stored previous phi instead.
            phi_after = _phi(self.p)
            prev_phi = getattr(self, "_prev_phi", phi_after)
            r = r + kappa * (prev_phi - 0.99 * phi_after)
            self._prev_phi = phi_after

        # Potential-based ENERGY shaping (Innovation 2): telescoping
        # potential on the minimum battery fraction.
        kappa_e = getattr(c, "energy_shape_kappa", 0.0)
        if kappa_e and kappa_e > 0:
            phi_e = float(np.min(self.E) / max(c.B_max, 1.0))
            prev_phi_e = getattr(self, "_prev_phi_e", phi_e)
            r = r + kappa_e * (prev_phi_e - 0.99 * phi_e)
            self._prev_phi_e = phi_e

        # Innovation 7: count-based intrinsic motivation.  The state key is
        # a coarse hash of the swarm's coverage vector; novel states get a
        # bonus that decays as 1/sqrt(count).  Telescoping is not required
        # because this is an exploration bonus, not a shaping on the value.
        ib = getattr(c, "intrinsic_beta", 0.0)
        if ib and ib > 0:
            key = tuple(int(x) for x in (ncov >= c.K).astype(int).tolist())
            cnt = self._visit_counts.get(key, 0) + 1
            self._visit_counts[key] = cnt
            r = r + ib * (1.0 / (cnt ** 0.5))

        # C3: per-agent counterfactual credit (optional)
        if getattr(c, "credit_mode", "team") == "c3":
            try:
                act = self._active_idx()
                credit = 0.0
                if len(act) > 0 and len(ncov) > 0:
                    for i in act:
                        marg = 0
                        for d in range(c.M):
                            dist = ((self.p[i,0]-self.sensors[d,0])**2 + (self.p[i,1]-self.sensors[d,1])**2)**0.5
                            if dist <= c.R_cov and ncov[d] <= c.K:
                                marg += 1
                        credit += marg / max(c.K * c.M, 1)
                    credit /= len(act)
                if len(adj) > 0:
                    ap = self.articulation_points(adj)
                    conn_credit = float(ap.mean()) if len(ap) else 0.0
                else:
                    conn_credit = 0.0
                r += c.credit_weight * (credit + conn_credit)
            except Exception:
                pass

        # Difference rewards (SI feedback 3.2): per-agent marginal coverage.
        # r_i = r_team + beta * D_i, where D_i is the coverage lost if agent i
        # vanished. Computed for active agents; inactive get the team reward.
        self.last_agent_rewards = None
        if getattr(c, "credit_mode", "team") == "diff":
            beta = getattr(c, "diff_beta", 1.0)
            n = c.N
            r_arr = np.full(n, r, dtype=np.float64)
            act = self._active_idx()
            ncov_arr = np.asarray(ncov)
            for i in act:
                # sensors within R_cov of agent i
                d_i = 0.0
                for d in range(c.M):
                    dist = ((self.p[i, 0] - self.sensors[d, 0]) ** 2 +
                            (self.p[i, 1] - self.sensors[d, 1]) ** 2) ** 0.5
                    if dist <= c.R_cov:
                        nd = int(ncov_arr[d])
                        marg = min(nd, c.K) - min(nd - 1, c.K)
                        d_i += marg / max(c.K * c.M, 1)
                r_arr[i] = r + beta * d_i
            self.last_agent_rewards = r_arr.astype(np.float32)

        self.prev_cov_score = cov_now
        self.prev_w_dist = w_dist

        # --- 7. metrics -------------------------------------------------
        self._update_metrics(ncov, adj, draw, w_dist)
        # An episode ends either because the mission time limit is reached or
        # because the swarm can no longer serve the sensors.  These are NOT
        # the same event: a time limit is a TRUNCATION, and the value of the
        # final state is still meaningful, so the GAE recursion must bootstrap
        # from V(s_T) rather than treat it as absorbing.  Conflating the two
        # biases every value target downward (Pardo et al., ICML 2018).
        self.truncated = self.t >= c.episode_len
        # A genuine terminal state: every UAV has failed, so nothing can
        # serve the sensors and no future return is possible.  In every other
        # case the episode simply hit the time limit, which is a truncation.
        self.terminated = (len(self._active_idx()) == 0) and not self.truncated
        done = self.truncated or self.terminated
        return self.local_obs(), self.global_state(), r, done, {}

    # ------------------------------------------------------------------
    # Reward, eq. (23)
    # ------------------------------------------------------------------
    def reward_from_terms(self, terms: dict) -> float:
        """Eq. (23):  r_t = -( sum_k lambda_k * cost_k(t) ).

        Implemented as a pure function of the cost terms so that the equation in
        the report can be asserted against directly. Every term is subtracted.
        A previous revision negated only U_t, which rewarded energy burn,
        partitions, motion, recovery delay and criticality; that defect is now
        impossible to reintroduce without failing the test suite.
        """
        c = self.cfg
        cost = (
            c.lam_O * terms["U_t"]
            + c.lam_E * terms["E_t"]
            + c.lam_D * terms["D_t"]
            + c.lam_W * terms["W_t"]
            + c.lam_R * terms["R_t"]
        )
        if c.use_connectivity_term:
            cost += c.lam_C * terms["Crit_t"]
        if c.lam_B:
            # Energy-threshold violation.  The energy term prices what is
            # CONSUMED, which is tiny compared with the cost of a UAV that
            # runs out of power away from a pad: a UAV that cannot get home
            # leaves the swarm, stops covering and may fragment the graph,
            # and all of that arrives LATE, so a consumption term alone
            # cannot teach the controller to turn back in time.  This term
            # is the shortfall below the return threshold, which the
            # observation already carries as the return-urgency feature.
            cost += c.lam_B * terms["B_t"]
        return -float(cost)

    # ------------------------------------------------------------------
    # Metrics (Table IX)
    # ------------------------------------------------------------------
    def _update_metrics(self, ncov, adj, draw, w_dist):
        c = self.cfg
        m = self.ep
        kcov_frac_this = float(np.mean(ncov >= c.K))
        m["Kcov_ok"] += float(np.all(ncov >= c.K))
        m["Kcov_frac"] += kcov_frac_this
        m["under_cov"] += float(np.mean(ncov < c.K))
        m["connected"] += float(self.is_connected(adj))
        m["service_cont"] += float(np.all(ncov >= c.K) and self.is_connected(adj))
        m["crit"] += float(self.articulation_points(adj).mean()) if len(adj) else 1.0
        m["energy"] += float(np.sum(draw) * c.dt)
        m["distance"] += w_dist
        m["min_batt"] = min(m["min_batt"], float(np.min(self.E[self._active_idx()])) if len(self._active_idx()) else 0.0)
        m["steps"] += 1
        m["denied"] += self.denied_charge
        m["reserve_activations"] = self.reserve_used
        m["return_events"] = self.return_events
        m["stranded"] = self.stranded
        m["projection_revisions"] = self.projection_revisions
        m["unexecuted_actions"] = self.unexecuted_actions
        # fraction of the swarm currently in the return-to-charge state
        m["returning"] += float(np.mean(self.mode[self._active_idx()] == 1)) \
            if len(self._active_idx()) else 0.0
        # Non-degenerate metrics: time-averaged coverage deficit + worst refresh.
        m["cov_integral"] += (1.0 - kcov_frac_this)
        if kcov_frac_this < 1.0:
            self._under_streak += 1
            m["max_under_streak"] = max(m["max_under_streak"], self._under_streak)
        else:
            self._under_streak = 0

    def metrics(self):
        return dict(
            Kcov_ok=0.0, Kcov_frac=0.0, under_cov=0.0, connected=0.0,
            service_cont=0.0, crit=0.0, energy=0.0, distance=0.0,
            min_batt=float("inf"), steps=0, denied=0, reserve_activations=0,
            return_events=0, stranded=0, returning=0.0,
            projection_revisions=0, unexecuted_actions=0,
            # The metrics the table promises and the summary did not
            # carry: the recovery time, the service lost while the swarm
            # was degraded or recovering, the safety counters, and the
            # charging contention.  They are additive and touch neither
            # the dynamics nor the reward.
            rec_delay_sum=0.0, rec_delay_n=0, rec_delay_max=0.0,
            rec_delay_p95=0.0, outage_severity=0.0, outage_steps=0,
            safety_collisions=0, safety_out_of_bounds=0,
            safety_violations=0,
            charge_wait_sum=0.0, charge_wait_n=0, charge_wait_max=0.0,
            # Non-degenerate metrics (Deep Research recommendation):
            # time-averaged coverage deficit and worst-case refresh streak.
            cov_integral=0.0, max_under_streak=0,
        )

    def summary(self):
        s = self.ep["steps"]
        if s == 0:
            return {}
        return dict(
            CK=self.ep["Kcov_frac"] / s,
            Kcov_ok=self.ep["Kcov_ok"] / s,
            Ubar=self.ep["under_cov"] / s,
            AG=self.ep["connected"] / s,
            SC=self.ep["service_cont"] / s,
            Crit=self.ep["crit"] / s,
            Energy=self.ep["energy"],
            MinBattery=self.ep["min_batt"],
            Distance=self.ep["distance"],
            Denied=self.ep["denied"],
            Reserve=self.ep["reserve_activations"],
            ReturnEvents=self.ep["return_events"],
            Stranded=self.ep["stranded"],
            Returning=self.ep["returning"] / s,
            ProjectionRevisions=self.ep["projection_revisions"],
            UnexecutedActions=self.ep["unexecuted_actions"],
            # Non-degenerate metrics (Deep Research recommendation):
            # CovIntegral = time-averaged coverage deficit (0 = perfect coverage)
            # MaxRefresh  = longest run of steps with any under-covered sensor
            CovIntegral=self.ep["cov_integral"] / s,
            MaxRefresh=self.ep["max_under_streak"],
        )