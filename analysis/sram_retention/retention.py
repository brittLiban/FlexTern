"""
EXP-0009 leakage model.

  P_active(T)            = provisioned_MB x leak25[level] x temp_factor(V_nom, T)                       (B1, B2)
  P_mode(T)              = P_active(T) x ratio(mode, T)                                                  (B4)
  ratio(LIGHT_SLEEP)     = 0.90                                                                          (R01)
  ratio(DEEP_SLEEP_NOM)  = 1 - periphery_share                                                           (B3)
  ratio(RETENTION_VRET)  = (1 - periphery_share) x [I(V_ret,T) V_ret] / [I(V_nom,T) V_nom]  (ASAP7, R13)  (B4)
  ratio(BODY_BIAS)       = 0.013  (R05 bound; FD-SOI only)

Temperature factors come from the ASAP7 cell table (T2); the model interpolates log-linearly between simulated points.
All leakage is for the ARRAY+PERIPHERY macro boundary; nothing here is a chip figure.
"""

from __future__ import annotations

import math

import retention_evidence as R

MODES = ("ACTIVE_STANDBY", "LIGHT_SLEEP", "DEEP_SLEEP_NOMINAL", "RETENTION_VRET", "BODY_BIAS_BOUND")
RETAINS = {"ACTIVE_STANDBY": True, "LIGHT_SLEEP": True, "DEEP_SLEEP_NOMINAL": True, "RETENTION_VRET": True, "BODY_BIAS_BOUND": True, "SHUTDOWN": False}
_TEMPS = sorted({t for (_, t) in R.ASAP7_CELL_A})
_VOLTS = sorted({v for (v, _) in R.ASAP7_CELL_A})


def _interp_log(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]; y0, y1 = math.log(ys[i - 1]), math.log(ys[i])
            return math.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return ys[-1]


def cell_current(v: float, temp_c: float) -> float:
    """ASAP7 TT 6T hold leakage current (A) at supply v and temperature, log-interpolated in T then V (R13)."""
    v = min(max(v, _VOLTS[0]), _VOLTS[-1])
    cols = []
    for vv in _VOLTS:
        cols.append(_interp_log(temp_c, _TEMPS, [R.ASAP7_CELL_A[(vv, t)] for t in _TEMPS]))
    return _interp_log(v, _VOLTS, cols)


def temp_factor(temp_c: float, v: float = R.ASAP7_VNOM) -> float:
    """Leakage multiplier relative to 25 C at supply v (T2 modeled, B2)."""
    return cell_current(v, temp_c) / cell_current(v, 25)


def array_power_ratio(v_ret: float, temp_c: float) -> float:
    """Array leakage POWER at V_ret over array leakage power at V_nom, same temperature (R13)."""
    return (cell_current(v_ret, temp_c) * v_ret) / (cell_current(R.ASAP7_VNOM, temp_c) * R.ASAP7_VNOM)


def active_mw_per_mb(level: str, temp_c: float) -> float:
    return R.ACTIVE_LEAK_25C_MW_PER_MB[level] * temp_factor(temp_c)


def mode_ratio(mode: str, temp_c: float, level: str = "planning", v_ret: float = R.V_RET_PLANNING) -> float:
    ps = R.PERIPHERY_SHARE[level]
    if mode == "ACTIVE_STANDBY":
        return 1.0
    if mode == "LIGHT_SLEEP":
        return R.LIGHT_SLEEP_RATIO
    if mode == "DEEP_SLEEP_NOMINAL":
        return 1.0 - ps
    if mode == "RETENTION_VRET":
        return (1.0 - ps) * array_power_ratio(v_ret, temp_c)
    if mode == "BODY_BIAS_BOUND":
        return R.BODY_BIAS_RATIO
    if mode == "SHUTDOWN":
        raise ValueError("SHUTDOWN loses data and is not a retention mode")
    raise KeyError(mode)


def mode_mw_per_mb(mode: str, level: str, temp_c: float, v_ret: float = R.V_RET_PLANNING) -> float:
    return active_mw_per_mb(level, temp_c) * mode_ratio(mode, temp_c, level, v_ret)


# ----------------------------------------------------------------------------- tables
def leakage_temperature_rows() -> list[dict]:
    rows = []
    for level in ("low", "planning", "high"):
        for t in R.TEMPS_C + (105,):
            rows.append({"level": level, "temp_c": t, "temp_factor_vs_25C_at_Vnom": round(temp_factor(t), 3), "temp_factor_vs_25C_at_Vret": round(temp_factor(t, R.V_RET_PLANNING), 3),
                         "active_standby_mW_per_MB": round(active_mw_per_mb(level, t), 4),
                         "light_sleep_mW_per_MB": round(mode_mw_per_mb("LIGHT_SLEEP", level, t), 4),
                         "deep_sleep_nominal_mW_per_MB": round(mode_mw_per_mb("DEEP_SLEEP_NOMINAL", level, t), 4),
                         "retention_vret_mW_per_MB": round(mode_mw_per_mb("RETENTION_VRET", level, t), 4),
                         "retention_vret_ratio": round(mode_ratio("RETENTION_VRET", t, level), 4),
                         "body_bias_bound_mW_per_MB": round(mode_mw_per_mb("BODY_BIAS_BOUND", level, t), 5),
                         "label": "B1 T1 anchors at 25 C; temperature scaling T2 MODELED (ASAP7); ratios B4"})
    return rows


def retention_voltage_rows() -> list[dict]:
    rows = []
    for t in R.TEMPS_C + (105,):
        for v in R.RETENTION_VOLTAGES + (R.ASAP7_VNOM,):
            rows.append({"temp_c": t, "v_ret": v, "cell_current_pA": round(cell_current(v, t) * 1e12, 2), "cell_power_pW": round(cell_current(v, t) * v * 1e12, 2),
                         "array_power_ratio_vs_Vnom": round(array_power_ratio(v, t), 4), "array_mW_per_MB": round(cell_current(v, t) * v * 8e6 * 1e3, 4),
                         "label": "ASAP7 TT 6T SPICE (R13, T2); hold SNM in hold_snm.csv"})
    return rows
