"""
EXP-0009 retention policies and memory-subsystem power model.

Resident system, per token period Tp = 1/R:
  E_dyn        = weight bits x 0.4405 pJ/bit                         (EXP-0008 planning, B11)
  leakage      = time-weighted: active MB-seconds x P_act + retained MB-seconds x P_ret
  E_wake       = wakes per token x E_wake(domain)                     (wake.py, B7)
  stall        = per-token latency added when a wake is not hidden   (wake.py, B6)
Policies:
  P0  all active (no retention)
  Pk  rolling window of k active layers (k = 1, 2, 4, 8); every layer wakes once per token (paced or burst timing)
  PS  session-active: all layers active while generating, whole model retained between sessions
  PT  timeout: whole model retained after X idle seconds, woken on the next token (burst compute)
LPDDR streaming (B10): E_dyn = bits x 4.0 pJ/bit; background = busy fraction x IDD2N + rest x IDD2P during generation;
self-refresh (temperature-dependent) between sessions when the gap >= 1 s, power-down otherwise.
Head and KV are external in both systems and added identically (system_reconnect).
"""

from __future__ import annotations

import math

import retention_evidence as R
import retention as RT
import wake as WK
import evidence as E8                      # EXP-0008 constants (LPDDR, dynamic energy)
from model import resident_pj_per_bit      # EXP-0008 resident energy per bit

E_RES_PJ = resident_pj_per_bit("planning")["total_pj_per_bit"]   # 0.4405 (B11)
E_LP_PJ = E8.LPDDR_PJ_PER_BIT["planning"]                         # 4.0 (B10)


def lpddr_background_mw(weight_bytes: float, R_tok: float) -> tuple[float, float]:
    """(busy fraction, background mW) during generation at R tok/s, one x16 channel at 70% (EXP-0008 A13)."""
    busy = min(1.0, weight_bytes * R_tok / (E8.LPDDR_CHANNEL_GBPS * 1e9 * 0.7))
    return busy, busy * E8.LPDDR_BACKGROUND_MW["idle_standby"] + (1 - busy) * E8.LPDDR_BACKGROUND_MW["power_down"]


def lpddr_self_refresh_mw(temp_c: float) -> float:
    lo, hi = E8.LPDDR_BACKGROUND_MW["self_refresh_25C"], E8.LPDDR_BACKGROUND_MW["self_refresh_85C"]
    f = min(max((temp_c - 25) / 60.0, 0.0), 1.0)
    return math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo)))


def lpddr_idle_mw(gap_s: float, temp_c: float) -> float:
    return lpddr_self_refresh_mw(temp_c) if gap_s >= R.SELF_REFRESH_GAP_S else E8.LPDDR_BACKGROUND_MW["power_down"]


class Scenario:
    """One model x bits x level x temperature x retention mode; holds the capacities and the per-MB powers."""

    def __init__(self, sched: dict, level: str = "planning", temp_c: float = 25, mode: str = "RETENTION_VRET", v_ret: float = R.V_RET_PLANNING,
                 engine_bw: float = R.ENGINE_BW_BPS["target"], wake_level: str | None = None, leak_mw_per_mb: float | None = None, wake_uj_per_mb: float | None = None,
                 ratio_override: float | None = None, lpddr_bg_scale: float = 1.0):
        self.s = sched; self.level = level; self.temp_c = temp_c; self.mode = mode; self.v_ret = v_ret; self.engine_bw = engine_bw
        self.ratio_override = ratio_override; self.lpddr_bg_scale = lpddr_bg_scale
        self.wake_level = wake_level or level
        self.L = sched["L"]
        self.W = sched["weight_bytes"]; self.prov = sched["provisioned_bytes"]
        self.layer_prov = sched["layer_provisioned_bytes"]; self.layer_bytes = sched["layer_bytes"]
        self.act_mw_mb = leak_mw_per_mb if leak_mw_per_mb is not None else RT.active_mw_per_mb(level, temp_c)
        self.ratio = ratio_override if ratio_override is not None else RT.mode_ratio(mode, temp_c, level, v_ret)
        self.ret_mw_mb = self.act_mw_mb * self.ratio
        self.e_dyn = self.W * 8 * E_RES_PJ * 1e-12            # J per token
        self.e_reload = self.W * 8 * (E_LP_PJ + E8.SRAM_MACRO_FJ_PER_BIT["planning"] / 1000 * E8.SRAM_WRITE_OVER_READ) * 1e-12   # B14 (EXP-0008 cold start)
        self.p_shutdown = self.prov / 1e6 * self.act_mw_mb * R.SHUTDOWN_RATIO * 1e-3                                             # residual switch leakage, data LOST
        self.e_lp = self.W * 8 * E_LP_PJ * 1e-12
        # wake energy per MB depends on whether the array rail swings (RETENTION_VRET) or only the periphery is restored
        if wake_uj_per_mb is not None:
            self.wake_uj_mb = wake_uj_per_mb
        elif mode == "RETENTION_VRET":
            self.wake_uj_mb = {"low": R.WAKE_UJ_PER_MB["planning"], "planning": R.WAKE_UJ_PER_MB_ARRAY_SWING, "high": R.WAKE_UJ_PER_MB_ARRAY_SWING * 2}[self.wake_level]
        else:
            self.wake_uj_mb = R.WAKE_UJ_PER_MB[self.wake_level]

    # --- leakage power of the whole model in a given state (W)
    def p_all_active(self) -> float:
        return self.prov / 1e6 * self.act_mw_mb * 1e-3

    def p_all_retained(self, temp_c: float | None = None) -> float:
        """Whole-model retention power; optionally at a different (idle) temperature than the scenario's."""
        if temp_c is None or temp_c == self.temp_c:
            return self.prov / 1e6 * self.ret_mw_mb * 1e-3
        ratio = self.ratio_override if self.ratio_override is not None else RT.mode_ratio(self.mode, temp_c, self.level, self.v_ret)
        return self.prov / 1e6 * RT.active_mw_per_mb(self.level, temp_c) * ratio * 1e-3

    def p_all_active_at(self, temp_c: float | None = None) -> float:
        if temp_c is None or temp_c == self.temp_c:
            return self.p_all_active()
        return self.prov / 1e6 * RT.active_mw_per_mb(self.level, temp_c) * 1e-3

    def wake_e(self, domain_bytes: float) -> float:
        return WK.wake_energy_j(domain_bytes, self.wake_level, uj_per_mb=self.wake_uj_mb)

    def wake_t(self, domain_bytes: float) -> float:
        return WK.wake_latency_s(domain_bytes, self.wake_level)

    def t_layer_burst(self, i: int) -> float:
        return self.layer_bytes[i] / self.engine_bw

    def t_compute(self) -> float:
        return self.W / self.engine_bw

    # --- policies: return dict of W (average power) and per-token quantities for continuous generation at R tok/s
    def P0(self, R_tok: float) -> dict:
        p_leak = self.p_all_active()
        return self._pack("P0", R_tok, p_leak, 0.0, 0.0, 1.0)

    def window(self, k: int, R_tok: float, pacing: str = "paced") -> dict:
        Tp = 1.0 / R_tok
        k = min(k, self.L)
        if pacing == "paced":
            t_layer = Tp / self.L
            act_frac = k / self.L                                   # capacity-time fraction active
        else:
            t_layer = self.t_compute() / self.L                     # average burst layer time
            act_frac = min(k * t_layer / Tp, k / self.L)
        p_leak = act_frac * self.p_all_active() + (1 - act_frac) * self.p_all_retained()
        e_wake = sum(self.wake_e(b) for b in self.layer_prov)       # every layer wakes once per token
        lead = (k - 1) * t_layer
        stall = sum(max(0.0, self.wake_t(b) - lead) for b in self.layer_prov)   # s per token
        d = self._pack(f"P{k}" + ("" if pacing == "paced" else "-burst"), R_tok, p_leak, e_wake, stall, act_frac)
        d["wakes_per_token"] = self.L; d["lead_time_us"] = lead * 1e6; d["layer_time_us"] = t_layer * 1e6
        d["wake_latency_layer_us"] = self.wake_t(max(self.layer_prov)) * 1e6
        d["wake_slack_us"] = (lead - self.wake_t(max(self.layer_prov))) * 1e6
        return d

    def timeout(self, X: float, R_tok: float) -> dict:
        """Burst compute each token, then whole-model retention after X idle seconds."""
        Tp = 1.0 / R_tok
        tc = self.t_compute()
        gap = max(0.0, Tp - tc)
        if gap > X:
            act_frac = (tc + X) / Tp
            e_wake = self.wake_e(self.prov); stall = self.wake_t(self.prov); wakes = 1
        else:
            act_frac = 1.0; e_wake = 0.0; stall = 0.0; wakes = 0
        p_leak = act_frac * self.p_all_active() + (1 - act_frac) * self.p_all_retained()
        d = self._pack(f"PT{X * 1e3:g}ms", R_tok, p_leak, e_wake, stall, act_frac)
        d["wakes_per_token"] = wakes; d["wake_latency_model_us"] = self.wake_t(self.prov) * 1e6
        return d

    def session(self, N: int, R_tok: float, gap_s: float, in_session: str = "P0", k: int = 2, idle_temp_c: float | None = None, idle_mode: str = "retention") -> dict:
        """Average power over one session of N tokens at R tok/s followed by gap_s of idle (whole model retained).
        idle_temp_c lets the idle period run cooler than the generating period (B13); default = same temperature.
        idle_mode = "shutdown" prices the PX alternative (B14): array off (data lost), one full reload per session."""
        t_gen = N / R_tok
        it = self.temp_c if idle_temp_c is None else idle_temp_c
        base = self.P0(R_tok) if in_session == "P0" else self.window(k, R_tok)
        if idle_mode == "shutdown":
            e_session = base["resident_total_W"] * t_gen + self.e_reload
            e_idle = self.p_shutdown * gap_s
        else:
            e_session = base["resident_total_W"] * t_gen + self.wake_e(self.prov)          # one whole-model wake per session
            e_idle = self.p_all_retained(it) * gap_s
        lp_session = base["lpddr_total_W"] * t_gen
        lp_idle = lpddr_idle_mw(gap_s, it) * 1e-3 * gap_s
        T = t_gen + gap_s
        return {"policy": (f"PS({in_session})" if idle_mode == "retention" else f"PX({in_session})"), "idle_mode": idle_mode, "session_tokens": N, "tok_s": R_tok, "idle_gap_s": gap_s, "session_time_s": t_gen, "idle_temp_c": it,
                "resident_avg_mW": (e_session + e_idle) / T * 1e3, "lpddr_avg_mW": (lp_session + lp_idle) / T * 1e3,
                "resident_session_mJ": e_session * 1e3, "resident_idle_mJ": e_idle * 1e3, "session_wake_mJ": (self.wake_e(self.prov) if idle_mode == "retention" else self.e_reload) * 1e3,
                "lpddr_session_mJ": lp_session * 1e3, "lpddr_idle_mJ": lp_idle * 1e3, "lpddr_idle_mode": "self-refresh" if gap_s >= R.SELF_REFRESH_GAP_S else "power-down",
                "lpddr_over_resident": ((lp_session + lp_idle) / (e_session + e_idle)) if (e_session + e_idle) > 0 else float("inf"),
                "first_token_stall_us": self.wake_t(self.prov) * 1e6 if idle_mode == "retention" else self.W / (R.RELOAD_GBPS[1] * 1e9) * 1e6}

    def _pack(self, name: str, R_tok: float, p_leak: float, e_wake: float, stall: float, act_frac: float) -> dict:
        p_dyn = self.e_dyn * R_tok; p_wake = e_wake * R_tok
        p_res = p_dyn + p_leak + p_wake
        busy, bg = lpddr_background_mw(self.W, R_tok)
        bg *= self.lpddr_bg_scale                                 # B15 sensitivity only (1.0 in planning)
        p_lp = self.e_lp * R_tok + bg * 1e-3
        return {"policy": name, "tok_s": R_tok, "level": self.level, "temp_c": self.temp_c, "mode": self.mode, "v_ret": self.v_ret,
                "active_capacity_fraction": act_frac, "active_leak_mW_per_MB": self.act_mw_mb, "retention_mW_per_MB": self.ret_mw_mb, "retention_ratio": self.ratio,
                "resident_dynamic_W": p_dyn, "resident_leakage_W": p_leak, "resident_wake_W": p_wake, "resident_total_W": p_res,
                "resident_dynamic_mJ_per_token": self.e_dyn * 1e3, "transition_mJ_per_token": e_wake * 1e3, "stall_us_per_token": stall * 1e6,
                "lpddr_dynamic_W": self.e_lp * R_tok, "lpddr_background_W": bg * 1e-3, "lpddr_total_W": p_lp, "lpddr_channel_busy": busy,
                "lpddr_over_resident": p_lp / p_res if p_res > 0 else float("inf")}

    def duty(self, d: float, R_active: float, policy: str = "PS", k: int = 2, idle_temp_c: float | None = None, lpddr_idle: str = "best") -> dict:
        """Average power at duty cycle d (fraction of time generating at R_active); idle = whole model retained except P0.
        idle_temp_c (B13): temperature of the idle periods; default = scenario temperature."""
        it = self.temp_c if idle_temp_c is None else idle_temp_c
        if policy == "P0":
            base = self.P0(R_active); idle = self.p_all_active_at(it)
        elif policy == "PS":
            base = self.P0(R_active); idle = self.p_all_retained(it)
        else:
            base = self.window(k, R_active); idle = self.p_all_retained(it)
        p_res = d * base["resident_total_W"] + (1 - d) * idle
        if lpddr_idle == "power_down":                                                            # EXP-0008 convention (reproduction only)
            lp_idle = E8.LPDDR_BACKGROUND_MW["power_down"] * 1e-3
        else:
            lp_idle = min(E8.LPDDR_BACKGROUND_MW["power_down"], lpddr_self_refresh_mw(it)) * 1e-3  # DRAM's best idle mode at the idle temperature (B10)
        p_lp = d * base["lpddr_total_W"] + (1 - d) * lp_idle
        return {"policy": policy if policy != "window" else f"P{k}+S", "duty_cycle": d, "active_tok_s": R_active, "idle_temp_c": it, "lpddr_idle_mW": lp_idle * 1e3, "resident_avg_mW": p_res * 1e3, "lpddr_avg_mW": p_lp * 1e3,
                "resident_idle_mW": idle * 1e3, "resident_static_share": (d * base["resident_leakage_W"] + (1 - d) * idle) / p_res if p_res > 0 else 0.0,
                "lpddr_over_resident": p_lp / p_res if p_res > 0 else float("inf")}
