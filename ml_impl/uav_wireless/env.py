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

                                                                            
                     
                                                                            
A_HOVER, A_PX, A_NX, A_PY, A_NY, A_CHARGE = range(6)
N_ACTIONS = 6
ACTION_VECTORS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]]
)


@dataclass
class SimConfig:
              
    N: int = 5                                 
    N_reserve: int = 1
    M: int = 20                                
    side: float = 80.0                                 
    R_cov: float = 35.0                            
    R_comm: float = 40.0                                                
    z0: float = 50.0                                
    K: int = 2                                                
    dt: float = 0.2                                 
    v_max: float = 5.0
    d_safe: float = 5.0
    B_max: float = 30_000.0          
    P_hover: float = 50.0            
    P_move: float = 75.0             
    P_elec: float = 5.0                                
    P_tx: float = 0.1                                                    
    P_tx_sensor: float = 0.02                                          
    charge_rate: float = 0.0002                              
    episode_len: int = 1_000                                          
    n_pads: int = 2                                 
    pad_service_radius: float = 15.0
    E_reserve: float = 2_000.0                       
    failure_enabled: bool = False
    failure_time: int = -1                       
    credit_mode: str = "team"                            
    credit_weight: float = 0.10
    diff_beta: float = 1.00                                            
    randomize_layout: bool = False                                           
    adversarial_layout: bool = False                                       
    ego_sensors: bool = False                                                          
    use_critical_anchor: bool = False                                                           
    n_ego: int = 3                                                                
    use_w_penalty: bool = True                                               
    dist_shape_kappa: float = 0.0                                                 
    energy_shape_kappa: float = 0.0                                              
    intrinsic_beta: float = 0.0                                              
                      
    lam_O: float = 1.00                           
    lam_E: float = 0.10                               
                                                                            
                                                                     
                                                                     
                                                                           
    lam_D: float = 2.00                             
    lam_W: float = 0.05                                 
    lam_R: float = 0.30                           
    lam_C: float = 0.50                                           
                                                                         
                                                                          
                                                                            
                                                                       
                                                                           
                                                                    
    lam_B: float = 0.0
                                            
    use_connectivity_term: bool = True
    use_reserve: bool = True
    use_charging_arbitration: bool = True
    use_action_mask: bool = True
                                                                             
                                                                            
                                                                  
                                                                         
                                            
    use_joint_projection: bool = True
                                                                             
                                                                          
                                                                         
                                                                      
                                                                         
    strict_energy_mask: bool = True
    mask_horizon: int = 3                                                    
                                                                            
                                                                           
                                                                           
                                                                     
    start_clustered: bool = False
                                                                             
                                                                            
                                                                             
                                                                          
                                                                            
                                                                          
                                                                            
                                                                            
             
    use_agent_id: bool = False
                                                                            
                                                                     
                                                                          
                                                                    
                                                                            
                                                                            
                                                
    map_grid: int = 0
                                                                          
                                                                          
                                                                         
                                                               
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
    base = 7 + 4 * 4 + 7                                                      
    base += cfg.N if cfg.use_agent_id else 0
    base += cfg.map_grid ** 2 if cfg.map_grid else 0
    base += 3 if cfg.use_pad_bearing else 0
    if getattr(cfg, "ego_sensors", False):
        base += getattr(cfg, "n_ego", 3) * 3 + 2                                              
    if getattr(cfg, "use_critical_anchor", False):
        base += 1                                                               
    return base


def state_dim(cfg: SimConfig) -> int:
    """Length of the centralised critic input for this configuration (Eq. 3)."""
    return 2 * cfg.N + 3 * cfg.N + 2 * cfg.M + cfg.n_pads + 3


class UAVIoTEnv:
    """Cooperative persistent-coverage Dec-POMDP."""

                                                                  
    NB_SLOTS = 4

                                                               
                                                                 
                                                            
                                                      
                                                              
    OBS_DIM = 7 + 4 * NB_SLOTS + 7                

                                                          
                                                                                
                                                                   
                                                                               
                                                                    
                                                                               
                                                                
    STATE_DIM = 2 * 5 + 3 * 5 + 2 * 20 + 2 + 3                          

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._episode_idx = 0
        self.radio = WirelessModel(z0=cfg.z0, p_tx=cfg.P_tx,
                                    p_tx_sensor=cfg.P_tx_sensor)
        self.obs_dim = obs_dim(cfg)
        self.state_dim = state_dim(cfg)
        self.reset()

                                                                        
           
                                                                        
    def reset(self):
        c = self.cfg
        if getattr(c, "randomize_layout", False):
            self.rng = np.random.default_rng(int(c.seed) * 1000003 + self._episode_idx * 999983 + 17)
        self.t = 0
                                                                              
                                                                             
                                                                            
                                                                            
                                                                               
        self.p = self._connected_initial_positions()
        self.v = np.zeros((c.N, 2))
        self.E = self.rng.uniform(0.7, 1.0, size=c.N) * c.B_max
        self.failed = np.zeros(c.N, dtype=bool)
        self.mode = np.zeros(c.N, dtype=int)                                       
        self.pad_assigned = np.full(c.N, -1, dtype=int)
        self.pad_queue = np.zeros(c.n_pads, dtype=int)
        self.pad_occupied_by = np.full(c.n_pads, -1, dtype=int)

        if getattr(c, "adversarial_layout", False):
                                                                        
                                                                
                                                  
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
        self.return_events = 0                                   
        self.stranded = 0                                                        
        self.reserve_activations = 0
        self.projection_revisions = 0                                           
        self.unexecuted_actions = 0                                            

        self.ep = self.metrics()
        self._under_streak = 0
        self.truncated = False
        self.terminated = False
        self._map_sum = self.coverage_map_summary()
        self.prev_metrics = dict(self.ep)
        return self.local_obs(), self.global_state()

                                                                        
                                           
                                                                        
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
        while len(pos) < c.N:                                                  
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
                                 
        d = np.linalg.norm(self.sensors[:, None, :] - self.p[act][None, :, :], axis=-1)
        within = d <= self.cfg.R_cov
                                                                               
                                                                               
                                                                               
        sinr = self._access_sinr_matrix(self.p[act], self.sensors, act)           
        rate = self.radio.rate_mbps(self.radio.snr_db(sinr))                      
        phy_ok = rate >= self.radio.rate_min_mbps                                 
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
        return p + c.P_elec + c.P_tx                                         

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

                                                                        
                                                     
                                                                        
    def action_mask(self, i, base=None):
        c = self.cfg
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[A_HOVER] = True

        act = self._active_idx()
        if i not in act:
                                                                
            return m
        others = act[act != i]
        step = c.v_max * c.dt
        dirs = [A_PX, A_NX, A_PY, A_NY]
        cands = np.clip(self.p[i] + np.array([ACTION_VECTORS[a] for a in dirs])
                         * step, 0, c.side)
                                                                               
                                                                               
                                                                            
                                                                            
                                                                          
                                          
        adj_c = self._candidate_adjacencies(i, cands, base=base) \
            if c.use_action_mask else None
                                                                            
                                                                 
        pad_energies = self._pad_energies(
            np.vstack([self.p[i][None, :], cands]))
        pad_now = float(pad_energies[0])

        for a_dir, a in enumerate(dirs):
            p_new = cands[a_dir]
            ok = True
                                                                 
            if np.any(p_new < 0) or np.any(p_new > c.side):
                ok = False
                        
            if ok and len(others):
                if np.min(np.linalg.norm(p_new - self.p[others], axis=1)) < c.d_safe:
                    ok = False
                                 
             
                                                                           
                                                                         
                                                                       
                                                                           
                                                                         
                                                                         
                                                                          
                                                                             
                                                                       
                      
             
                                                                          
                                                                    
                                                                       
            if ok:
                e_after = self.E[i] - c.P_move * c.dt
                if c.strict_energy_mask:
                    pad_new = float(pad_energies[a_dir + 1])
                    reserve = c.E_reserve
                    if e_after < pad_new + reserve and pad_new >= pad_now:
                        ok = False
                elif e_after <= 0:
                    ok = False
                                                                         
                                                                          
                                                                           
                                                                        
            if ok and c.use_action_mask:
                if not self._h_step_safe(i, p_new, c.mask_horizon,
                                         adj=adj_c[a_dir]):
                    ok = False
                elif self._is_articulation_move(i, p_new, adj=adj_c[a_dir]):
                    ok = False
            m[a] = ok

                                                                        
                              
         
                                                                      
                                                     
                                                                      
                                                                    
                                                                   
                                                                  
                                                                   
                                                   
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

                                                                               
        rows = self.radio.backhaul_snr_rows(p_act, act, pos_i, cands)          
        d_new = np.linalg.norm(p_act[None, :, :] - cands[:, None, :],
                               axis=-1)                                         

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
                self.v[i] = np.zeros(2)                              
                return 1
            self.p[i] = cur
                                                                           
                                                                        
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
                                                                          
                                                                         
                                                                    
                                                                     
        n = len(adj)
        pos_i = list(act).index(i)
                                                                            
                                                                          
                                                                             
                          
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
                                                                             
                                                                         
                row = np.concatenate([row, self.pad_bearing(i)])
            if c.map_grid:
                                                                           
                row = np.concatenate([row, self._map_sum])
            if c.use_agent_id:
                                                                           
                                                                          
                                                                           
                                                                           
                                            
                ident = np.zeros(c.N, dtype=np.float32)
                ident[i] = 1.0
                row = np.concatenate([row, ident])
            if getattr(c, "ego_sensors", False):
                                                                       
                                                              
                k = getattr(c, "n_ego", 3)
                deficient = [d for d in range(c.M) if ncov[d] < c.K]
                                             
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
            (self.p / c.side).flatten(),              
            self.E / c.B_max,                        
            self.mode / 2.0,                         
            self.failed.astype(float),               
            (self.sensors / c.side).flatten(),        
            (self.pad_occupied_by >= 0).astype(float),           
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

                                                                        
          
                                                                        
    def step(self, actions):
        c = self.cfg
        act_before = self._active_idx().copy()

                                                                         
        step = c.v_max * c.dt
                                                                            
                                                                        
                                            
        masks = self.action_masks()
        old_p = self.p.copy()
        masked_actions = 0
        for i in self._active_idx():
            a = int(actions[i])
            if a in (A_PX, A_NX, A_PY, A_NY) and not masks[i][a]:
                                                                       
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
                                                                
             
                                                                         
                                                                          
                                                                           
                                                                           
                                                                            
                                                                            
                                                                            
                                                                            
                                                                            
            if masks[i][a]:
                self.p[i] = p_new
            elif a in (A_PX, A_NX, A_PY, A_NY):
                pass                                                                  

                                                                        
                                                                            
                                                                      
                                                                             
                                                                          
                                                                      
                                                                      
                                                                           
                                                                       
                                                                          
                                 
        moved = [i for i in self._active_idx()
                 if not np.allclose(self.p[i], old_p[i])]
        if c.use_joint_projection and moved:
            self.projection_revisions += self._project_connected(old_p, moved)
                                                                          
                                                                           
                                                                           
                                                                      
        reverted = sum(1 for i in self._active_idx()
                       if not np.allclose(self.p[i], old_p[i]))
        self.unexecuted_actions += masked_actions + max(0, len(moved) - reverted)

                                                                         
        if self.failure_t >= 0 and self.t == self.failure_t:
            victim = act_before[self.rng.integers(len(act_before))]
            self.failed[victim] = True
            self.mode[victim] = -1
            self.recovered = False
            self.recovery_start = self.t
            self.failure_t = -1
            self.recovered = False
        elif c.use_reserve and self.reserve_used < c.N_reserve and not self.recovered:
                              
            self.reserve_used += 1
            self.failed = self.failed.copy()
                                                               
            dead = np.where(self.failed)[0]
            if len(dead):
                j = int(dead[0])
                self.p[j] = self.pads[self.rng.integers(len(self.pads))]
                self.E[j] = c.B_max
                self.failed[j] = False
                self.mode[j] = 0
                self.recovered = False

                                                                         
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

                                                                         
        draw = self.power_draw(actions)
        self.E = np.clip(self.E - draw * c.dt, 0, c.B_max)
        for i in range(c.N):
            if self.failed[i]:
                self.E[i] = 0.0

                                                                          
                                                                          
                                           
         
                                                                           
                                                                         
                                                                      
                                                                      
                                                                             
                                                                       
                                                                           
                                                                    
        for i in self._active_idx():
            thresh = self.return_threshold(i)
            if self.mode[i] == 0 and self.E[i] <= thresh:
                self.mode[i] = 1                                  
                self.return_events += 1
            elif self.mode[i] == 1 and self.E[i] >= thresh:
                self.mode[i] = 0                                        

                                                                           
                                                                     
                                                                        
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

                                                                         
        ncov = self.coverage_counts()
                                                                            
                                                                          
                                             
        self._map_sum = self.coverage_map_summary(ncov)
        _, adj, _ = self.comm_graph()
        connected_now = self.is_connected(adj)
                                                                                
                                                                                
                                                                          
                                                                               
                                                                      
        degraded_now = not connected_now
        ok_now = (float(np.mean(ncov >= c.K)) >= 1.0) and connected_now
        if not self.recovered and ok_now:
            self.recovered = True
        rec_delay = 0.0
        if self.recovery_start >= 0 and self.recovered:
            rec_delay = min((self.t - self.recovery_start) / c.episode_len, 1.0)

                                                                         
         
                                                                            
                                                                             
                                                                            
                                                                        
                                                                         
                                                                  
        U_t = float(np.mean(ncov < c.K))
        E_cons = float(np.sum(draw) * c.dt / (c.N * c.B_max))
        D_t = 1.0 if degraded_now else 0.0
        w_dist = float(np.linalg.norm(self.v, axis=1).sum() * c.dt) if act_before.size else 0.0
                                                                              
                                                                               
                                                                 
        cov_now = float(np.mean(ncov >= c.K))
        newly_covered = max(cov_now - self.prev_cov_score, 0.0)
        if getattr(c, "use_w_penalty", True) and w_dist > 1e-9:
            W_t = float(np.clip(1.0 - newly_covered / (w_dist + 1e-9), 0.0, 1.0))
        else:
            W_t = 0.0
        crit = float(self.articulation_points(adj).mean()) if len(adj) else 1.0
        R_rec = rec_delay

                                                                        
                                                                           
                                                                         
                                                                           
                                                                        
        act_now = self._active_idx()
        if len(act_now):
            shortfalls = [max(0.0, self.return_threshold(i) - self.E[i])
                          / max(self.cfg.B_max, 1.0) for i in act_now]
            B_t = float(np.mean(shortfalls))
        else:
            B_t = 0.0

                                                                            
                                                                                
                                                           
        self.last_cost = dict(
            U_t=U_t, E_t=E_cons, D_t=D_t, W_t=W_t, R_t=R_rec, Crit_t=crit,
            B_t=B_t,
        )
        r = self.reward_from_terms(self.last_cost)

                                                                                
        kappa = getattr(c, "dist_shape_kappa", 0.0)
        if kappa and kappa > 0:
            def _phi(positions):
                                                                                   
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
                                                                                                        
                                                
            phi_after = _phi(self.p)
            prev_phi = getattr(self, "_prev_phi", phi_after)
            r = r + kappa * (prev_phi - 0.99 * phi_after)
            self._prev_phi = phi_after

                                                                    
                                                    
        kappa_e = getattr(c, "energy_shape_kappa", 0.0)
        if kappa_e and kappa_e > 0:
            phi_e = float(np.min(self.E) / max(c.B_max, 1.0))
            prev_phi_e = getattr(self, "_prev_phi_e", phi_e)
            r = r + kappa_e * (prev_phi_e - 0.99 * phi_e)
            self._prev_phi_e = phi_e

                                                                           
                                                                          
                                                                          
                                                                           
        ib = getattr(c, "intrinsic_beta", 0.0)
        if ib and ib > 0:
            key = tuple(int(x) for x in (ncov >= c.K).astype(int).tolist())
            cnt = self._visit_counts.get(key, 0) + 1
            self._visit_counts[key] = cnt
            r = r + ib * (1.0 / (cnt ** 0.5))

                                                        
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

                                                                            
                                                                              
                                                                             
        self.last_agent_rewards = None
        if getattr(c, "credit_mode", "team") == "diff":
            beta = getattr(c, "diff_beta", 1.0)
            n = c.N
            r_arr = np.full(n, r, dtype=np.float64)
            act = self._active_idx()
            ncov_arr = np.asarray(ncov)
            for i in act:
                                                 
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

                                                                          
        self._update_metrics(ncov, adj, draw, w_dist)
                                                                             
                                                                           
                                                                            
                                                                              
                                                                            
                                                                       
        self.truncated = self.t >= c.episode_len
                                                                        
                                                                             
                                                                            
        self.terminated = (len(self._active_idx()) == 0) and not self.truncated
        done = self.truncated or self.terminated
        return self.local_obs(), self.global_state(), r, done, {}

                                                                        
                      
                                                                        
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
                                                                         
                                                                          
                                                                           
                                                                          
                                                                       
                                                                          
                                                                    
                                                                        
            cost += c.lam_B * terms["B_t"]
        return -float(cost)

                                                                        
                        
                                                                        
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
                                                                       
        m["returning"] += float(np.mean(self.mode[self._active_idx()] == 1)) \
            if len(self._active_idx()) else 0.0
                                                                                 
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
                                                                    
                                                                        
                                                                      
                                                                       
                                          
            rec_delay_sum=0.0, rec_delay_n=0, rec_delay_max=0.0,
            rec_delay_p95=0.0, outage_severity=0.0, outage_steps=0,
            safety_collisions=0, safety_out_of_bounds=0,
            safety_violations=0,
            charge_wait_sum=0.0, charge_wait_n=0, charge_wait_max=0.0,
                                                                    
                                                                           
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
                                                                    
                                                                                 
                                                                              
            CovIntegral=self.ep["cov_integral"] / s,
            MaxRefresh=self.ep["max_under_streak"],
        )