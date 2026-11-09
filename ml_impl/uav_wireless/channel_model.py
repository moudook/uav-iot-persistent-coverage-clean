"""Information channel model calibrated against OMNeT++/INET.

Source of truth: Simulation/omnet_uav2/pair_pdr_freespace.json and
pair_pdr_outage.json (INET, 50 m altitude, 20 dBm, -85 dBm sensitivity,
-94 dBm noise floor, LogNormalShadowing sigma = 6 dB).

At the mission scale (80 m, 8 UAVs) this model gives PDR = 1.0000: the
wireless layer is reliable and is NOT the binding constraint.  The binding
constraints are energy and coverage geometry.  The model's job is to certify
that, and to state the range (1762 m) beyond which it stops being true.
"""
import math

F_HZ = 2412000000.0
TX_DBM = 20.0
SENS_DBM = -85.0
SIGMA_DB = 6.0
KNEE_M = 1762.0


def path_loss_db(d, f=F_HZ):
    return 20.0 * math.log10(max(d, 1.0)) + 20.0 * math.log10(f) - 147.55


def pdr(d, sigma=SIGMA_DB):
    """Delivery probability for a link of length d (metres)."""
    pl = path_loss_db(d)
    rx_mean = TX_DBM - pl
    z = (rx_mean - SENS_DBM) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def reliable(d, threshold=0.99):
    return pdr(d) >= threshold
