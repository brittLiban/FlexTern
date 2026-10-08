"""
EXP-0009 retention policies and memory-subsystem power model (B7 and B14 corrected 2026-10-08, EXP-0009A).

Resident system, per token period Tp = 1/R:
  E_dyn        = weight bits x 0.4405 pJ/bit                         (EXP-0008 planning, B11)
  leakage      = time-weighted: active MB-seconds x P_act + retained MB-seconds x P_ret
  E_wake       = wakes per token x E_wake(domain)                     (wake.py, B7 corrected)
  stall        = per-token latency added when a wake is not hidden   (wake.py, B6)
Policies:
  P0  all active (no retention)
  Pk  rolling window of k active layers (k = 1, 2, 4, 8); every layer wakes once per token (paced or burst timing)
  PS  session-active: all layers active while generating, whole model retained between sessions
  PT  timeout: whole model retained after X idle seconds, woken on the next token (burst compute)
  PX  shutdown-and-reload between sessions (data lost): PX-DRAM (B16) or PX-FLASH (B17); the backing store's standby,
      reload energy and reload latency are charged; PX-UNBACKED reproduces the superseded EXP-0009 row.
LPDDR streaming (B10): E_dyn = bits x 4.0 pJ/bit; background = busy fraction x IDD2N + rest x IDD2P during generation;
self-refresh (temperature-dependent) between sessions when the gap >= 1 s, power-down otherwise; optionally DRAM off with a
flash reload for long gaps (symmetric to PX-FLASH).
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
BACKINGS = ("dram", "flash", "unbacked_superseded")


def lpddr_background_mw(weight_bytes: float, R_tok: float) -> tuple[float, float]:
    """(busy fraction, background mW) during generation at R tok/s, one x16 channel at 70% (EXP-0008 A13)."""
    busy = min(1.0, weight_bytes * R_tok / (E8.LPDDR_CHANNEL_GBPS * 1e9 * 0.7))
    return busy, busy * E8.LPDDR_BACKGROUND_MW["idle_standby"] + (1 - busy) * E8.LPDDR_BACKGROUND_MW["power_down"]


def lpddr_self_refresh_mw(temp_c: float) -> float:
    lo, hi = E8.LPDDR_BACKGROUND_MW["self_refresh_25C"], E8.LPDDR_BACKGROUND_MW["self_refresh_85C"]
    f = min(max((temp_c - 25) / 60.0, 0.0), 1.0)
    return math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo)))


def lpddr_idle_mw(gap_s: float, temp_c: float) -> float:
    """Data-retaining DRAM idle power for a gap: self-refresh (>= 1 s) or power-down (B10)."""
    return lpddr_self_refresh_mw(temp_c) if gap_s >= R.SELF_REFRESH_GAP_S else E8.LPDDR_BACKGROUND_MW["power_down"]


class Scenario:
    """One model x bits x level x temperature x retention mode; holds the capacities and the per-MB powers.

    wake_model: "corrected" (B7 component model, EXP-0009A) or "exp0009_original" (superseded 899976e values, reproduction only).
    vret_supply: "separate_rail" (planning; array power at V_ret) or "linear" (B18 sensitivity; current ratio, energy drawn at V_nom).
    backing_level: LOW / PLANNING / HIGH for the flash-backed reload (B17)."""

    def __init__(self, sched: dict, level: str = "planning", temp_c: float = 25, mode: str = "RETENTION_VRET", v_ret: float = R.V_RET_PLANNING,
                 engine_bw: float = R.ENGINE_BW_BPS["target"], wake_level: str | None = None, leak_mw_per_mb: float | None = None, wake_uj_per_mb: float | None = None,
                 ratio_override: float | None = None, lpddr_bg_scale: float = 1.0, wake_model: str = "corrected", vret_supply: str = "separate_rail",
                 backing_level: str | None = None):
        self.s = sched; self.level = level; self.temp_c = temp_c; self.mode = mode; self.v_ret = v_ret; self.engine_bw = engine_bw
        self.ratio_override = ratio_override; self.lpddr_bg_scale = lpddr_bg_scale
        self.wake_level = wake_level or level; self.wake_model = wake_model; self.vret_supply = vret_supply
        self.backing_level = backing_level or level
        if vret_supply not in R.VRET_SUPPLY_OPTIONS:
            raise ValueError(vret_supply)
        self.L = sched["L"]
        self.W = sched["weight_bytes"]; self.prov = sched["provisioned_bytes"]
        self.layer_prov = sched["layer_provisioned_bytes"]; self.layer_bytes = sched["layer_bytes"]
        self.act_mw_mb = leak_mw_per_mb if leak_mw_per_mb is not None else RT.active_mw_per_mb(level, temp_c)
        self.ratio = ratio_override if ratio_override is not None else self._ratio_at(temp_c)
        self.ret_mw_mb = self.act_mw_mb * self.ratio
        self.e_dyn = self.W * 8 * E_RES_PJ * 1e-12            # J per token
        self.e_lp = self.W * 8 * E_LP_PJ * 1e-12
        # --- B14 superseded reload (EXP-0009 899976e): LPDDR path energy + macro write only; kept for the PX-UNBACKED reproduction row
        self.e_reload_superseded = self.W * 8 * (E_LP_PJ + E8.SRAM_MACRO_FJ_PER_BIT["planning"] / 1000 * E8.SRAM_WRITE_OVER_READ) * 1e-12
        # --- B16 DRAM-backed reload: LPDDR read path + full SRAM write path; one x16 channel at 70%
        self.e_reload_dram = self.W * 8 * (E_LP_PJ + R.SRAM_WRITE_PATH_PJ_PER_BIT) * 1e-12
        self.t_reload_dram = self.W / R.RELOAD_DRAM_BPS
        # --- B17 flash-backed reload: UFS device read energy + SRAM write path; UFS sequential bandwidth; standby while idle
        self.e_reload_flash = self.W * 8 * (R.FLASH_READ_PJ_PER_BIT[self.backing_level] + R.SRAM_WRITE_PATH_PJ_PER_BIT) * 1e-12
        self.t_reload_flash = self.W / R.FLASH_RELOAD_BPS[self.backing_level]
        self.p_flash_standby = R.FLASH_STANDBY_MW[self.backing_level] * 1e-3
        # --- symmetric baseline option: DRAM off for long gaps, reloaded from flash (flash read + LPDDR write path)
        self.e_reload_lpddr_from_flash = self.W * 8 * (R.FLASH_READ_PJ_PER_BIT[self.backing_level] + E_LP_PJ) * 1e-12
        self.p_shutdown = self.prov / 1e6 * self.act_mw_mb * R.SHUTDOWN_RATIO * 1e-3                                             # residual switch leakage, data LOST
        # --- wake energy per MB (J): corrected component model (B7) or the superseded EXP-0009 figure
        if wake_uj_per_mb is not None:
            self.wake_j_mb = wake_uj_per_mb * 1e-6
        elif wake_model == "exp0009_original":
            self.wake_j_mb = WK.exp0009_original_j_per_mb(mode, self.wake_level)
        elif wake_model == "corrected":
            self.wake_j_mb = WK.wake_j_per_mb(mode, self.wake_level, R.V_ACTIVE, v_ret)
        else:
            raise ValueError(wake_model)
        self.wake_uj_mb = self.wake_j_mb * 1e6                  # kept for CSV columns (uJ/MB)

    def _ratio_at(self, temp_c: float) -> float:
        r = RT.mode_ratio(self.mode, temp_c, self.level, self.v_ret)
        if self.vret_supply == "linear" and self.mode == "RETENTION_VRET":
            r *= R.ASAP7_VNOM / self.v_ret                       # B18: energy drawn at V_nom -> current ratio
        return r

    # --- leakage power of the whole model in a given state (W)
    def p_all_active(self) -> float:
        return self.prov / 1e6 * self.act_mw_mb * 1e-3

    def p_all_retained(self, temp_c: float | None = None) -> float:
        """Whole-model retention power; optionally at a different (idle) temperature than the scenario's."""
        if temp_c is None or temp_c == self.temp_c:
            return self.prov / 1e6 * self.ret_mw_mb * 1e-3
        ratio = self.ratio_override if self.ratio_override is not None else self._ratio_at(temp_c)
        return self.prov / 1e6 * RT.active_mw_per_mb(self.level, temp_c) * ratio * 1e-3

    def p_all_active_at(self, temp_c: float | None = None) -> float:
        if temp_c is None or temp_c == self.temp_c:
            return self.p_all_active()
        return self.prov / 1e6 * RT.active_mw_per_mb(self.level, temp_c) * 1e-3

    def p_shutdown_at(self, temp_c: float | None = None) -> float:
        if temp_c is None or temp_c == self.temp_c:
            return self.p_shutdown
        return self.prov / 1e6 * RT.active_mw_per_mb(self.level, temp_c) * R.SHUTDOWN_RATIO * 1e-3

    def wake_e(self, domain_bytes: float) -> float:
        return WK.wake_energy_j(domain_bytes, self.wake_level, j_per_mb=self.wake_j_mb)

    def wake_t(self, domain_bytes: float) -> float:
        return WK.wake_latency_s(domain_bytes, self.wake_level)

    def t_layer_burst(self, i: int) -> float:
        return self.layer_bytes[i] / self.engine_bw

    def t_compute(self) -> float:
        return self.W / self.engine_bw

    # --- reload accounting (B14/B16/B17): energy (J), first-token latency (s), backing standby while idle and during the session (W)
    def reload(self, backing: str, gap_s: float, idle_temp_c: float | None = None, view: str = "weights_only") -> dict:
        it = self.temp_c if idle_temp_c is None else idle_temp_c
        if backing == "dram":
            p_idle = lpddr_idle_mw(gap_s, it) * 1e-3
            p_sess = E8.LPDDR_BACKGROUND_MW["power_down"] * 1e-3 if view == "weights_only" else 0.0
            return {"backing": "dram", "e_reload": self.e_reload_dram, "t_reload": self.t_reload_dram, "p_backing_idle": p_idle, "p_backing_session": p_sess,
                    "backing_idle_mode": "self-refresh" if gap_s >= R.SELF_REFRESH_GAP_S else "power-down"}
        if backing == "flash":
            return {"backing": "flash", "e_reload": self.e_reload_flash, "t_reload": self.t_reload_flash, "p_backing_idle": self.p_flash_standby, "p_backing_session": 0.0,
                    "backing_idle_mode": {"low": "powered down (host re-init)", "planning": "UFS sleep", "high": "UFS hibernate"}[self.backing_level]}
        if backing == "unbacked_superseded":
            return {"backing": "unbacked_superseded", "e_reload": self.e_reload_superseded, "t_reload": self.W / (R.RELOAD_GBPS[1] * 1e9), "p_backing_idle": 0.0, "p_backing_session": 0.0,
                    "backing_idle_mode": "none charged (SUPERSEDED EXP-0009 row)"}
        raise ValueError(backing)

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

    def session(self, N: int, R_tok: float, gap_s: float, in_session: str = "P0", k: int = 2, idle_temp_c: float | None = None, idle_mode: str = "retention",
                backing: str = "flash", view: str = "weights_only", baseline_idle: str = "retain") -> dict:
        """Average power over one session of N tokens at R tok/s followed by gap_s of idle.
        idle_mode = "retention": whole model retained between sessions (PS); one whole-model wake per session when the session runs
          P0 (all layers must be active); zero extra wakes when it runs a rolling window (the window's per-token layer wakes already
          cover the first token: transitions are counted exactly once).
        idle_mode = "shutdown": array off (data lost), reloaded from `backing` at the next session (PX-DRAM B16 / PX-FLASH B17 /
          PX-UNBACKED superseded); backing standby charged while idle (and during the session in the weights-only view for DRAM).
        idle_temp_c lets the idle period run cooler than the generating period (B13); default = same temperature.
        baseline_idle = "retain": LPDDR in its data-retaining idle mode; "shutdown_flash": DRAM off, reloaded from flash (symmetric)."""
        t_gen = N / R_tok
        it = self.temp_c if idle_temp_c is None else idle_temp_c
        base = self.P0(R_tok) if in_session == "P0" else self.window(k, R_tok)
        e_gen = base["resident_total_W"] * t_gen
        if idle_mode == "shutdown":
            rl = self.reload(backing, gap_s, it, view)
            e_transition = rl["e_reload"]
            e_session = e_gen + e_transition + rl["p_backing_session"] * t_gen
            e_idle = (self.p_shutdown_at(it) + rl["p_backing_idle"]) * gap_s
            first_token = rl["t_reload"]
            name = {"dram": "PX-DRAM", "flash": "PX-FLASH", "unbacked_superseded": "PX-UNBACKED(superseded)"}[backing] + f"({in_session})"
            backing_idle_mode = rl["backing_idle_mode"]; p_backing_idle = rl["p_backing_idle"]
        elif idle_mode == "retention":
            e_transition = self.wake_e(self.prov) if in_session == "P0" else 0.0          # exactly one whole-model wake per session under P0
            e_session = e_gen + e_transition
            e_idle = self.p_all_retained(it) * gap_s
            first_token = self.wake_t(self.prov) if in_session == "P0" else 0.0
            name = f"PS({in_session})"; backing_idle_mode = "n/a (retained)"; p_backing_idle = 0.0
        else:
            raise ValueError(idle_mode)
        lp_session = base["lpddr_total_W"] * t_gen
        if baseline_idle == "retain":
            lp_idle = lpddr_idle_mw(gap_s, it) * 1e-3 * gap_s; lp_mode = "self-refresh" if gap_s >= R.SELF_REFRESH_GAP_S else "power-down"; lp_first = 0.0
        elif baseline_idle == "shutdown_flash":
            lp_idle = self.p_flash_standby * gap_s; lp_session += self.e_reload_lpddr_from_flash; lp_mode = "DRAM off; flash reload"; lp_first = self.t_reload_flash
        else:
            raise ValueError(baseline_idle)
        T = t_gen + gap_s
        return {"policy": name, "idle_mode": idle_mode, "backing": backing if idle_mode == "shutdown" else "", "view": view, "session_tokens": N, "tok_s": R_tok, "idle_gap_s": gap_s, "session_time_s": t_gen, "idle_temp_c": it,
                "resident_avg_mW": (e_session + e_idle) / T * 1e3, "lpddr_avg_mW": (lp_session + lp_idle) / T * 1e3,
                "resident_session_mJ": e_session * 1e3, "resident_idle_mJ": e_idle * 1e3, "session_wake_mJ": e_transition * 1e3,
                "backing_idle_mW": p_backing_idle * 1e3, "backing_idle_mode": backing_idle_mode,
                "lpddr_session_mJ": lp_session * 1e3, "lpddr_idle_mJ": lp_idle * 1e3, "lpddr_idle_mode": lp_mode, "baseline_idle": baseline_idle,
                "lpddr_over_resident": ((lp_session + lp_idle) / (e_session + e_idle)) if (e_session + e_idle) > 0 else float("inf"),
                "first_token_stall_us": first_token * 1e6, "lpddr_first_token_stall_us": lp_first * 1e6}

    def shutdown_break_even_gap_s(self, backing: str, N: int = 64, R_tok: float = 25, in_session: str = "P0", idle_temp_c: float | None = None, view: str = "weights_only") -> float:
        """Idle gap above which PX(backing) averages less resident energy than PS (retention), solved numerically on session();
        inf if PX never wins below 1e5 s, 0 if it wins already at 1e-3 s."""
        def diff(g):
            return self.session(N, R_tok, g, in_session, idle_temp_c=idle_temp_c, idle_mode="retention")["resident_avg_mW"] - self.session(N, R_tok, g, in_session, idle_temp_c=idle_temp_c, idle_mode="shutdown", backing=backing, view=view)["resident_avg_mW"]
        lo, hi = 1e-3, 1e5
        if diff(hi) < 0:
            return float("inf")
        if diff(lo) > 0:
            return 0.0
        for _ in range(80):
            m = math.sqrt(lo * hi)
            if diff(m) > 0:
                hi = m
            else:
                lo = m
        return hi

    def shutdown_break_even_gap_closed_form_s(self, backing: str, in_session: str = "P0", idle_temp_c: float | None = None, gap_hint_s: float = 10.0) -> float:
        """Closed form from the idle-period terms (valid when the backing's session-time standby is zero, i.e. flash or the system view):
        gap* = (E_reload - E_session_wake_PS) / (P_ret - P_residual - P_backing_idle); the DRAM idle mode is taken at gap_hint_s."""
        it = self.temp_c if idle_temp_c is None else idle_temp_c
        rl = self.reload(backing, gap_hint_s, it, "system")
        num = rl["e_reload"] - (self.wake_e(self.prov) if in_session == "P0" else 0.0)
        den = self.p_all_retained(it) - self.p_shutdown_at(it) - rl["p_backing_idle"]
        return float("inf") if den <= 0 else num / den

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
