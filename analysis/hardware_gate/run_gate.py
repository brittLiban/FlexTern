"""
EXP-0005 driver: runs the analytical models, collects the synthesis table, and writes
results/hardware_gate/summary.md plus environment.json and the primary comparison table.

Everything labeled.  MEASURED = RTL simulation and generic synthesis outputs produced here;
ANALYTICAL = derived from the measured workload shapes and the engine cycle laws.
No physical quantity (frequency, bandwidth, energy, area) is measured anywhere.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, _HERE)
os.chdir(ROOT)

import traffic_model as tm  # noqa: E402
import workload_mapping as wm  # noqa: E402
from reference import accumulator_bits, int4_accumulator_bits  # noqa: E402

OUT = os.path.join("results", "hardware_gate")


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def tool_line(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return ((r.stdout or r.stderr).strip().splitlines() or ["?"])[0]
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e.__class__.__name__})"


def read_csv(name: str) -> list[dict]:
    p = os.path.join(OUT, name)
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.isfile(p) else []


def fmt_mb(b) -> str:
    return f"{float(b) / 1e6:,.1f} MB"


def main() -> int:
    ws = wm.load_workloads()
    rows, totals = wm.mapping_rows(ws)
    wm.write_csv(os.path.join(OUT, "workload_mapping.csv"), rows)
    wm.write_csv(os.path.join(OUT, "workload_totals.csv"), totals)
    wm.write_csv(os.path.join(OUT, "bandwidth_requirements.csv"), wm.bandwidth_rows())
    traffic = []
    for slug, w in ws.items():
        for nb in (16, 64, 256, 1024):
            for bits in (2, 4):
                traffic.append(tm.per_token(w, nb, bits, buffered=True))
            traffic.append(tm.per_token(w, nb, 2, buffered=False))
    tm.write_csv(os.path.join(OUT, "traffic_comparison.csv"), traffic)
    tm.write_csv(os.path.join(OUT, "capacity_scaling.csv"), tm.capacity_rows(ws))
    tm.write_csv(os.path.join(OUT, "system_reconnect.csv"), tm.reconnect_rows(ws))

    synth = {r["point"]: r for r in read_csv("synthesis.csv")}
    runs = json.load(open(os.path.join(OUT, "synth", "runs.json"), encoding="utf-8")) if os.path.isfile(os.path.join(OUT, "synth", "runs.json")) else {}

    # ---------------------------------------------------------------- environment
    ivdir = os.environ.get("FLEXTERN_IVERILOG_DIR") or r"C:\iverilog\bin"
    env = {"date": date.today().isoformat(), "commit": git_head(), "os": platform.platform(), "python": sys.version.split()[0],
           "iverilog": tool_line([os.path.join(ivdir, "iverilog.exe"), "-V"]),
           "yosys_yowasp": tool_line([os.path.join(os.path.dirname(sys.executable), "Scripts", "yowasp-yosys.exe"), "-V"]),
           "yosys_wsl": tool_line(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", "yosys -V"]),
           "synthesis_runs": runs, "fpga_target": "none (founder has no FPGA; physical timing, utilization and bandwidth UNMEASURED by design)"}
    json.dump(env, open(os.path.join(OUT, "environment.json"), "w", encoding="utf-8"), indent=1)

    # ---------------------------------------------------------------- primary comparison table (Falcon-E 1B representative tensors)
    f1b = ws["falcon-e-1b"]
    prim = []
    for t in wm.tensors(f1b):
        for eng, cfg, point in (("streaming", (64, 1), "stream_b64_r1"), ("streaming", (256, 1), "stream_b256_r1"),
                                ("near-memory", (64, 16), "bank_b16"), ("near-memory", (256, 16), "bank_b16")):
            tr = tm.per_token(f1b, cfg[0] if eng == "near-memory" else 1)
            if eng == "streaming":
                cyc, util = wm.stream_cycles(t, *cfg); banks = 1; cap = ""; terms_pc = 4 * cfg[0] * cfg[1]
                glob = t["packed_bytes"]; loc = 0; inj = t["K"]; deliv = t["K"]; outb = t["N"] * tm.OUT_BYTES
                cells = synth.get(point, {}).get("total_generic_cells", ""); dff = synth.get(point, {}).get("dff_cells", "")
                status = "MEASURED cells; cycles ANALYTICAL from the RTL cycle law (II=1 verified in simulation)"
            else:
                cyc, util, cap = wm.nearmem_cycles(t, *cfg); banks = cfg[0]; terms_pc = 4 * cfg[0] * cfg[1]
                glob = 0; loc = t["packed_bytes"]; inj = t["K"]; deliv = t["K"] * cfg[0]; outb = t["N"] * tm.OUT_BYTES
                c1 = synth.get(point, {}).get("total_generic_cells", ""); d1 = synth.get(point, {}).get("dff_cells", "")
                cells = int(c1) * cfg[0] if c1 else ""; dff = int(d1) * cfg[0] if d1 else ""
                status = "cells = one synthesized bank x NB (collector measured separately); weights resident (preload untimed)"
            prim.append({"model": "falcon-e-1b", "tensor": t["module"], "K": t["K"], "N": t["N"], "terms": t["terms"],
                         "packed_weight_bytes": t["packed_bytes"], "engine": eng, "parallelism": f"{cfg}", "terms_per_cycle": terms_pc,
                         "banks": banks, "cycles": cyc, "utilization": round(util, 3),
                         "global_weight_bytes": glob, "local_weight_bytes": loc, "activation_injected_bytes": inj,
                         "activation_delivered_bytes": deliv, "output_bytes": outb, "partial_sum_bytes": 0,
                         "accumulator_bits": accumulator_bits(t["K"]), "memory_capacity_bytes_per_bank": cap,
                         "activation_buffer_bits_total": 8 * t["K"] * banks,
                         "synthesized_logic_cells": cells, "dff_cells": dff, "memory_bits_in_logic": 0, "status": status})
    tm.write_csv(os.path.join(OUT, "primary_comparison.csv"), prim)

    # ---------------------------------------------------------------- summary.md
    L = ["# EXP-0005 — Packed ternary consumer: streaming vs banked near-memory (summary)", "",
         f"Generated by `analysis/hardware_gate/run_gate.py` at commit {env['commit']} on {env['date']}. Labels: MEASURED (RTL simulation, generic synthesis), ANALYTICAL (cycle laws and byte counts from the measured shapes). **No physical quantity is measured: there is no FPGA target.** Required clocks are TARGET requirements, not achieved frequencies.", "",
         "## Packing interpretation (re-verified)", "",
         "`packed[r, c]` bits `[2i+1:2i]` = code of `W[i*(N/4) + r, c]`, code = q + 1 (00 = -1, 01 = 0, 10 = +1, 11 unused -> hardware contributes 0). One byte = four OUTPUT rows at ONE input position; consecutive bytes walk K. The engines consume this directly; no repack exists. Checkpoint slices (Falcon-E 1B k_proj/down_proj, Falcon-E 3B down_proj, BitNet 2B4T q_proj/down_proj) ran through both engines bit-identically to the Python reference (`tests/test_hardware_gate.py`).", "",
         "## Generic synthesis (MEASURED, technology-independent cells; not area/timing/power)", "",
         "| point | kind | terms/cycle | bytes/cycle | cells | cells/term | DFF | $mul | LTP (gates, no FF) | tool |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for p in wm.__dict__.get("_unused", []) or []:
        pass
    for name, r in synth.items():
        L.append(f"| {name} | {r['kind']} | {r['terms_per_cycle']} | {r['packed_bytes_per_cycle']} | {r['total_generic_cells']} | {r['cells_per_term']} | {r['dff_cells']} | {r['mul_inferred']} | {r['ltp_gates_noff']} | {r['yosys_version']} |")
    L += ["", "Flip-flops are the only state in every module (memories are ports): no N x K expansion exists anywhere. See `synthesis.csv` and `synth/*.txt`.", "",
          "## Required clock for bandwidth targets (TARGET / ANALYTICAL, UNMEASURED)", "",
          "| engine | config | packed bytes/cycle | 100 GB/s | 250 GB/s | 500 GB/s |", "|---|---|---:|---:|---:|---:|"]
    for r in wm.bandwidth_rows():
        L.append(f"| {r['engine']} | {r['config']} | {r['packed_bytes_per_cycle']} | {r['required_clock_MHz_for_100GBps']} MHz | {r['required_clock_MHz_for_250GBps']} MHz | {r['required_clock_MHz_for_500GBps']} MHz |")
    L += ["", "required_clock = target_weight_bandwidth / packed_bytes_per_cycle. Absolute line rate remains UNMEASURED.", "",
          "## Per-token cycles and utilization on the real workloads (ANALYTICAL from measured shapes)", "",
          "| model | engine | config | terms/cycle | cycles/token | utilization | capacity/bank (max) |", "|---|---|---|---:|---:|---:|---:|"]
    for r in totals:
        L.append(f"| {r['model']} | {r['engine']} | {r['config']} | {r['terms_per_cycle']} | {r['cycles_per_token']:,} | {r['utilization_token_weighted']} | {r['capacity_bytes_per_bank_max'] if r['capacity_bytes_per_bank_max'] == '' else fmt_mb(r['capacity_bytes_per_bank_max'])} |")
    L += ["", "## Four-level traffic per token, batch 1 (ANALYTICAL)", "",
          "| model | banks | bits | A load once | stream B global | nm B global | nm C local | act injected | act delivered (buffered) | outputs | nm global ÷ stream global |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in traffic:
        if r["bank_buffers"]:
            L.append(f"| {r['model']} | {r['banks']} | {r['weight_bits']} | {fmt_mb(r['stream_A_load_once'])} | {fmt_mb(r['stream_B_global_per_token'])} | {fmt_mb(r['nm_B_global_per_token'])} | {fmt_mb(r['nm_C_local_reads'])} | {fmt_mb(r['nm_D_activation_injected'])} | {fmt_mb(r['nm_D_activation_delivered'])} | {fmt_mb(r['nm_D_outputs'])} | {r['nm_global_over_stream_global']} |")
    lock = [r for r in traffic if not r["bank_buffers"] and r["banks"] == 64]
    L += ["", "Without bank-local activation buffers (lockstep re-broadcast once per row-quad pass) the delivered activation bytes equal the packed weight bytes: " +
          ", ".join(f"{r['model']} {fmt_mb(r['nm_D_activation_delivered'])}" for r in lock) + ". Bank buffers cost K x 8 bits per bank (13 KB at K = 13312).", "",
          "## Capacity (ANALYTICAL from measured sizes)", "", "| model | packed total | per layer | largest tensor | banks @64 KB | @256 KB | @1 MB | @4 MB | @16 MB | per bank @64 banks | @256 | @1024 |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    caps = tm.capacity_rows(ws)
    for slug in ws:
        cr = [r for r in caps if r["model"] == slug]
        byc = {r["bank_capacity_bytes"]: r for r in cr if r["bank_capacity_bytes"] != ""}
        byn = {r["bank_count"]: r for r in cr if r["bank_count"] != ""}
        L.append(f"| {slug} | {fmt_mb(cr[0]['packed_bytes_total'])} | {fmt_mb(cr[0]['packed_bytes_per_layer'])} | {cr[0]['largest_tensor']} ({fmt_mb(cr[0]['largest_tensor_bytes'])}) | "
                 + " | ".join(str(byc[c]['banks_needed_whole_model']) for c in tm.CAPACITIES) + " | "
                 + " | ".join(fmt_mb(byn[n]['capacity_needed_per_bank']) for n in (64, 256, 1024)) + " |")
    L += ["", "Whole-model residency (hundreds of MB) is required for the near-memory global-traffic result; per-layer residency reloads the full model every token and gains nothing. The memory technology that would hold it is NOT chosen here.", "",
          "## EXP-0003 / EXP-0004 reconnection (ANALYTICAL, 100 GB/s, batch 1; results/roofline and results/model_precision untouched)", "",
          "| model | ctx | scenario | ternary global (stream) | ternary local (nm) | act+out global (nm) | head | KV | global total stream | global total nm | reduction | limiter nm |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in tm.reconnect_rows(ws):
        if r["context"] in (2048, 8192, 32768):
            L.append(f"| {r['model']} | {r['context']} | {r['scenario']} | {fmt_mb(r['ternary_global_bytes_streaming'])} | {fmt_mb(r['ternary_local_bytes_near_memory'])} | {fmt_mb(r['activation_plus_output_global_bytes_near_memory'])} | {fmt_mb(r['head_bytes'])} | {fmt_mb(r['kv_bytes'])} | {fmt_mb(r['global_total_streaming'])} | {fmt_mb(r['global_total_near_memory'])} | {r['global_reduction_factor']}x | {r['limiter_near_memory']} |")
    L += ["", "Reading: near-memory removes the ternary weight stream from the global path; what remains is the head and the KV cache, which EXP-0004 showed can be ~4-bit but which this experiment does not consume. Global-byte reduction is not energy reduction and not end-to-end speedup.", "",
          "## Accumulator widths (format bound, INT8 activations)", "",
          "| K | ternary (bits) | INT4 weights (bits) |", "|---:|---:|---:|"]
    for k in (2048, 2560, 6912, 9216, 13312):
        L.append(f"| {k} | {accumulator_bits(k)} | {int4_accumulator_bits(k)} |")
    L += ["", "Primary comparison table: `primary_comparison.csv`. Full tables: `workload_mapping.csv`, `workload_totals.csv`, `traffic_comparison.csv`, `capacity_scaling.csv`, `bandwidth_requirements.csv`, `system_reconnect.csv`, `synthesis.csv`."]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("summary written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
