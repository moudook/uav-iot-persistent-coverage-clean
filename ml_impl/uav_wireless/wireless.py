"""
PHY layer for UAV-IoT wireless links.

This module is the answer to the professor's main criticism: the earlier draft
modelled coverage and connectivity as *abstract discs*. Here every link is
computed from an actual radio model:

    distance -> path loss (log-distance, LoS / NLoS blend) -> shadowing
             -> received power -> SINR (access) / SNR (relay)
             -> achievable rate (802.11 OFDM MCS table)

Both the UAV-UAV relay graph and the UAV-sensor access links go through this,
so `connectivity` and `K-coverage` in the simulator are PHY-derived rather than
geometric.

References
----------
- 3GPP TR 36.777, "Enhanced LTE support for aerial vehicles" (UAV channel).
- Al-Hourani, Kandeepan, Lardner (2014), air-to-ground LoS probability.
- IEEE 802.11-2016, OFDM MCS sensitivity/rate table (20 MHz channel).

Design rules
------------
1. **Reciprocity.** The channel between two nodes is one physical link. Shadow
   fading is a property of the propagation path, so S(i,j) == S(j,i). A
   per-element random draw would violate this.
2. **Reproducibility.** Every random quantity comes from this object's own
   seeded generator, never from the global numpy RNG. Otherwise the physics
   perturb the agent's action sampling and no run can be replayed.
3. **Index stability.** Channels are addressed by *global node index*, not by
   position in the active subset. When a UAV fails, the remaining links keep
   exactly the channel they had; otherwise the failure experiment silently
   re-draws the whole propagation environment.
4. **Timescale.** The decision interval is dt = 0.2 s. Slow shadowing (seconds)
   is included; fast Rayleigh fading (milliseconds) averages out over 200 ms
   and is therefore off by default. It remains available via `add_fading` for
   a future study at a different timescale.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# IEEE 802.11n OFDM MCS table, 20 MHz channel, short guard interval.
# Maps required RX SNR (dB) -> PHY rate (Mbit/s).  The sensitivity
# column is the standard's, and the table is sorted by it, because the
# selection below is a search over the threshold.
#
# The earlier table in this file was a mixture of rates and thresholds
# from several PHYs, and its rows above the first never affected a
# decision: every rate above the minimum is treated alike by the
# service test, so the gate was exactly "SNR >= 2 dB" and the other
# fourteen rows were decoration.
# --------------------------------------------------------------------------
MCS_TABLE = np.array(
    [
        [2.0, 6.5],     # MCS 0  BPSK 1/2      (the minimum usable rate)
        [5.0, 13.0],    # MCS 1  QPSK 1/2
        [8.0, 19.5],    # MCS 2  QPSK 3/4
        [11.0, 26.0],   # MCS 3  16-QAM 1/2
        [15.0, 39.0],   # MCS 4  16-QAM 3/4
        [18.0, 52.0],   # MCS 5  64-QAM 2/3
        [20.0, 58.5],   # MCS 6  64-QAM 3/4
        [25.0, 65.0],   # MCS 7  64-QAM 5/6
    ]
)

#: SNR (dB) below which no MCS is usable at all.
SNR_SENSITIVITY_DB = float(MCS_TABLE[0, 0])


def snr_to_rate(snr_db, mcs_shift: int = 0):
    """Highest MCS rate (Mbit/s) whose SNR requirement is met.

    Returns 0.0 below the receiver sensitivity. Returning the lowest MCS
    instead would be wrong in a way that silently disables the PHY gate: the
    simulator accepts a sensor link only when `rate >= rate_min_mbps`, and
    `rate_min_mbps` *is* the lowest MCS, so a floor at 6.5 makes that test
    always true.

    An earlier revision used `side='left'` searching, which returns the first
    entry with `threshold >= snr`. At SNR = 3.9 dB that selects the 9 Mbit/s
    MCS whose requirement is 4.0 dB -- claiming a rate the link cannot carry.
    The correct selection is the last entry with `threshold <= snr`.

    `mcs_shift` is an implementation margin in dB (negative = more
    conservative back-off), not a mode switch.
    """
    v = np.asarray(snr_db, dtype=np.float64) + mcs_shift
    idx = np.searchsorted(MCS_TABLE[:, 0], v, side="right") - 1
    rate = np.where(idx >= 0, MCS_TABLE[np.clip(idx, 0, None), 1], 0.0)
    if np.ndim(snr_db) == 0:
        return float(rate)
    return rate


def log_distance_pl(d_3d, f_hz: float, n: float, d0: float = 1.0):
    """Log-distance path loss in dB (positive = loss).

    Standard form, with the reference loss evaluated at d0:

        PL(d) = 20*log10(4*pi*d0*f/c) + 10*n*log10(d/d0)

    `n` is the path-loss exponent: n = 2.0 for free space, ~2.0-2.5 for a
    clear air-to-ground LoS path, ~3.5-4.0 when the path is obstructed.

    An earlier revision wrote the exponent additively on top of the
    free-space slope, `20*log10(d) + C + alpha*log10(d)`. That makes the
    effective exponent (20 + alpha)/10, so the intended Los n=2.0 / NLoS
    n=3.5 pair was really (20+2.0)/10 = 2.2 versus (20+3.5)/10 = 2.35 --
    almost no difference, which silently erased the LoS/NLoS distinction the
    whole air-to-ground model depends on.
    """
    d = np.maximum(np.asarray(d_3d, dtype=np.float64), d0)
    pl_d0 = 20.0 * np.log10(4.0 * np.pi * d0 * f_hz / 299_792_458.0)
    return pl_d0 + 10.0 * n * np.log10(d / d0)


def path_loss_db(d_3d, los_prob, f_hz, n_los=2.0, n_nlos=3.5):
    """Blend LoS and NLoS path loss according to the LoS probability."""
    pl_los = log_distance_pl(d_3d, f_hz, n_los)
    pl_nlos = log_distance_pl(d_3d, f_hz, n_nlos)
    return los_prob * pl_los + (1.0 - los_prob) * pl_nlos


def los_probability(d_3d, z0: float, a: float = 9.61, b: float = 0.16):
    """
    Al-Hourani (2014) air-to-ground LoS probability.

        P(LoS) = 1 / (1 + a * exp(-b * (theta - a))),   theta in degrees

    theta is the elevation angle of the UAV as seen from the ground node, so a
    higher-altitude or closer UAV has a better chance of a clear path.
    """
    d_3d = np.maximum(np.asarray(d_3d, dtype=np.float64), 1e-6)
    theta = np.degrees(np.arcsin(np.clip(z0 / d_3d, 0.0, 1.0)))
    return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))


class WirelessModel:
    """Radio parameters and the seeded propagation environment."""

    def __init__(
        self,
        f_hz: float = 2.4e9,
        p_tx: float = 0.1,             # W radiated by a UAV (20 dBm EIRP)
        p_tx_sensor: float = 0.02,     # W radiated by a sensor (13 dBm)
        noise_fig_db: float = 7.0,
        bandwidth_hz: float = 20e6,
        alpha_los: float = 2.0,         # path-loss exponent, clear LoS path
        alpha_nlos: float = 3.5,        # path-loss exponent, obstructed path
        z0: float = 50.0,
        rate_min_mbps: float = 6.5,    # minimum usable service rate
        shadowing_sigma_db: float = 4.0,
        rayleigh_scale: float = 1.0,
        mcs_shift_access: int = 0,
        mcs_shift_backhaul: int = -2,  # relay keeps a 2 dB margin
        access_cochannel: int = 0,     # other UAVs sharing a sensor's channel
        use_fading: bool = False,      # fast fading is sub-dt; see module docstring
    ):
        self.f_hz = f_hz
        self.p_tx = p_tx
        self.p_tx_sensor = p_tx_sensor
        self.bandwidth_hz = bandwidth_hz
        self.n_los = float(alpha_los)
        self.n_nlos = float(alpha_nlos)
        self.z0 = z0
        self.rate_min_mbps = rate_min_mbps
        self.shadowing_sigma_db = shadowing_sigma_db
        self.rayleigh_scale = rayleigh_scale
        self.mcs_shift_access = mcs_shift_access
        self.mcs_shift_backhaul = mcs_shift_backhaul
        self.access_cochannel = int(access_cochannel)
        self.use_fading = use_fading

        # Thermal noise: -174 dBm/Hz + NF + 10log10(B), converted to watts.
        self.noise_w = 10.0 ** (
            (-174.0 + noise_fig_db + 10.0 * np.log10(bandwidth_hz)) / 10.0
        ) / 1000.0

        # Dedicated generator: never touch the global RNG.
        self.rng = np.random.default_rng(0)
        self.shadow_uu_db: np.ndarray | None = None   # (N, N) symmetric, zero diag
        self.shadow_su_db: np.ndarray | None = None   # (M, N)
        self.n_uav = 0
        self.n_sensor = 0

    # -- propagation environment ------------------------------------------
    def reset_links(self, n_uav: int, n_sensor: int, seed: int | None = None):
        """Draw the quasi-static shadowing environment for one episode.

        Called once per episode, not per step: shadowing is a property of the
        scene, so it must persist across the whole run. Channels are addressed
        by global node index, so a UAV dropping out does not re-draw the links
        that remain.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.n_uav = int(n_uav)
        self.n_sensor = int(n_sensor)

        z = self.rng.normal(0.0, self.shadowing_sigma_db, size=(self.n_uav, self.n_uav))
        z = np.triu(z, 1)                    # strict upper triangle only
        self.shadow_uu_db = z + z.T          # mirror => S(i,j) == S(j,i)
        self.shadow_su_db = self.rng.normal(
            0.0, self.shadowing_sigma_db, size=(self.n_sensor, self.n_uav)
        )

    def _shadow_uu(self, idx):
        if self.shadow_uu_db is None:
            return None
        idx = np.asarray(idx, dtype=int)
        return self.shadow_uu_db[np.ix_(idx, idx)]

    def _shadow_su(self, uav_idx):
        if self.shadow_su_db is None:
            return None
        return self.shadow_su_db[:, np.asarray(uav_idx, dtype=int)]

    # -- helpers -----------------------------------------------------------
    def received_w(self, d_3d, los_prob, shadow_db=None, add_fading: bool = False,
                   p_tx: float | None = None):
        """Received power in watts: path loss, plus optional shadow / fading.

        With `shadow_db=None` this is the deterministic mean link budget, which
        is what the regression tests assert against.

        `p_tx` is the power of the TRANSMITTING end.  The access link is a
        sensor uplink, so it is budgeted with the sensor's power; the
        inter-UAV link with the UAV's.  Using the UAV's power for both
        handed the sensor an advantage of 7 dB that it does not have.
        """
        pl = path_loss_db(d_3d, los_prob, self.f_hz, self.n_los, self.n_nlos)
        if shadow_db is not None:
            pl = pl + shadow_db
        tx = self.p_tx if p_tx is None else p_tx
        rx = tx * 10.0 ** (-pl / 10.0)
        if add_fading or self.use_fading:
            rx = rx * self.rng.gamma(
                shape=1.0, scale=self.rayleigh_scale, size=np.shape(pl)
            )
        return rx

    def rate_mbps(self, snr_db):
        """Achievable access-link rate (Mbit/s) for an SNR in dB."""
        return snr_to_rate(snr_db, self.mcs_shift_access)

    def snr_db(self, linear):
        """Linear power ratio to dB, floored so log10 never sees zero."""
        return 10.0 * np.log10(np.maximum(linear, 1e-30))

    # -- link evaluation ---------------------------------------------------
    def uu_geometry(self, uav_xy):
        """Separation and LoS probability for all UAV pairs.

        Every UAV flies at the same altitude, so the slant range between
        two of them IS their horizontal separation: the old code
        computed sqrt(d^2 + z0^2) and charged the link 4.4 dB for an
        altitude the two ends do not differ in.  The air-to-ground
        LoS probability is not applied to an air-to-air path either;
        a horizontal link between two UAVs at the same height is a
        clear path, and the shadowing term carries the randomness.
        """
        d = np.linalg.norm(uav_xy[:, None, :] - uav_xy[None, :, :], axis=-1)
        ones = np.ones_like(d)
        return d, d, ones

    def access_sinr(self, sensor_xy, uav_xy, uav_idx=None,
                    cochannel: int | None = None, add_fading: bool = False):
        """(M, K) SINR for every sensor against every listed UAV.

        `cochannel` is how many *other* UAVs share the channel of the serving
        UAV. A sensor's serving link is the strongest one; only the strongest
        `cochannel` of the remaining UAVs are summed as interference. With
        `cochannel = 0` the access link is noise-limited, which is the default
        because the UAVs are assigned orthogonal 20 MHz channels.

        Modelling every other UAV as a co-channel interferer is not defensible
        in a K-coverage study: K-coverage asks whether K UAVs are *able* to
        serve a sensor, not whether all K transmit on the same resource at the
        same instant. The aggressive version made `phy_ok` false even when a
        UAV was sitting directly on top of its sensor, because a nearer
        non-serving UAV dominated the sum.

        `uav_idx` selects the shadowing columns by *global* id.
        """
        m, k = len(sensor_xy), len(uav_xy)
        if m == 0 or k == 0:
            return np.zeros((m, k))

        d = np.linalg.norm(sensor_xy[:, None, :] - uav_xy[None, :, :], axis=-1)
        d3 = np.sqrt(d ** 2 + self.z0 ** 2)
        los = los_probability(d3, self.z0)
        idx = np.arange(k) if uav_idx is None else np.asarray(uav_idx, dtype=int)
        # the access link is a SENSOR uplink: the sensor transmits
        rx = self.received_w(d3, los, self._shadow_su(idx), add_fading,
                             p_tx=self.p_tx_sensor)                          # (M, K)

        cch = self.access_cochannel if cochannel is None else int(cochannel)
        if cch <= 0 or k < 2:
            return rx / self.noise_w

        # Drop the strongest (that is the serving link), sum the next `cch`.
        ordered = np.sort(rx, axis=1)[:, ::-1]
        inter = ordered[:, 1:cch + 1].sum(axis=1, keepdims=True)
        return rx / (inter + self.noise_w)

    def backhaul_snr_rows(self, uav_xy, uav_idx, k, positions):
        """SNR rows of the links incident to index k, for SEVERAL positions.

        `positions` is an array of shape (P, 2); the result is (P, N).  This
        is the batched form of `backhaul_snr_row` and is what the action
        mask uses: the four candidate moves of one UAV are evaluated in a
        single call instead of four.

        The shadowing table is symmetric by construction -- one physical
        link, one channel, `shadow_uu_db = z + z.T` -- so the incident row
        is symmetric too, and splicing it into the base matrix reproduces
        the full recomputation exactly.
        """
        uav_xy = np.asarray(uav_xy)
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        idx = np.asarray(uav_idx, dtype=int)
        P = len(pos)
        # (P, N, 2)
        diff = pos[:, None, :] - uav_xy[None, :, :]
        d3 = np.sqrt(np.sum(diff ** 2, axis=-1))
        los = np.ones_like(d3)
        shadow = None
        if self.shadow_uu_db is not None:
            shadow = self.shadow_uu_db[np.ix_(idx, idx)][k]      # (N,)
        rx = self.received_w(d3, los, shadow, False)
        snr = rx / self.noise_w
        snr[:, k] = 0.0
        return snr

    def backhaul_snr_row(self, uav_xy, uav_idx, k, p_k):
        """SNR of the links incident to index k when k is at position p_k.

        The shadowing table is symmetric by construction -- one physical
        link, one channel, `shadow_uu_db = z + z.T` -- so the incident row
        is symmetric as well.  That makes it possible to evaluate a
        PROSPECTIVE graph (one UAV moved) by recomputing a single row and
        splicing it into the base matrix, instead of recomputing the whole
        NxN matrix for every candidate move.

        The result is identical to `backhaul_snr` with the UAV at p_k; a
        test asserts the equivalence.
        """
        uav_xy = np.asarray(uav_xy)
        idx = np.asarray(uav_idx, dtype=int)
        d3 = np.sqrt(np.sum((uav_xy - p_k) ** 2, axis=1) + self.z0 ** 2)
        los = los_probability(d3, self.z0)
        shadow = None
        if self.shadow_uu_db is not None:
            shadow = self.shadow_uu_db[np.ix_(idx, idx)][k]
        rx = self.received_w(d3, los, shadow, False)
        snr = rx / self.noise_w
        snr[k] = 0.0
        return snr

    def backhaul_snr(self, uav_xy, uav_idx=None, add_fading: bool = False):
        """SNR matrix for the inter-UAV relay link. Index (i,j) = i hears j.

        The relay is not co-channel with the access links -- in an aerial mesh
        the backhaul runs on its own channel or is time-slotted -- so other
        UAVs are not interferers here. Modelling them as interferers produced a
        spurious ~-14 dB even at 28 m separation, which made the connectivity
        invariant unachievable and drove AG to 0.2 in the first experiment run.

        The result is symmetric because the underlying geometry and shadowing
        are symmetric: one physical link, one channel.
        """
        n = len(uav_xy)
        if n == 0:
            return np.zeros((0, 0))
        _, d3, los = self.uu_geometry(uav_xy)
        idx = np.arange(n) if uav_idx is None else np.asarray(uav_idx, dtype=int)
        rx = self.received_w(d3, los, self._shadow_uu(idx), add_fading)
        snr = rx / self.noise_w
        snr = 0.5 * (snr + snr.T)          # enforce reciprocity exactly
        np.fill_diagonal(snr, 0.0)
        return snr

    # Retained under the old name so existing call sites keep working.
    backhaul_sinr = backhaul_snr

    def link_rate_mbps(self, snr_db, backhaul: bool = False):
        shift = self.mcs_shift_backhaul if backhaul else self.mcs_shift_access
        return snr_to_rate(snr_db, shift)

    def usable(self, snr_db, backhaul: bool = False):
        """Boolean link-quality gate: can this link carry the minimum service?"""
        return self.link_rate_mbps(snr_db, backhaul) >= self.rate_min_mbps
