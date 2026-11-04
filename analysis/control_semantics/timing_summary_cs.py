"""Roll up the short interleaved timing comparison of EXP-0010C (hold_L05 frames 0-2699) into results/control_semantics/timing_summary.json."""
import glob
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results" / "control_semantics"


def main():
    rows = {}
    for p in sorted(glob.glob(str(RES / "runs" / "hold_L05" / "*" / "replay_t*_timing_r*" / "summary.json"))):
        pp = Path(p)
        man = json.loads(pp.with_name("manifest.json").read_text())
        s = json.loads(pp.read_text())
        key = f"{pp.parents[1].name}_{pp.parent.name.split('_')[1]}"
        rows.setdefault(key, []).append({
            "run": pp.parent.name, "p50": s["latency_ms"]["p50"], "p95": s["latency_ms"]["p95"], "p99": s["latency_ms"]["p99"],
            "max": s["latency_ms"]["max"], "dropped_fraction": s["dropped_fraction"], "stage_ms_mean": s["stage_ms_mean"],
            "avg_busy_cores": s["resources"]["avg_busy_cores"], "clock_before_mhz": s["host_clock_before"].get("cur_mhz"),
            "host_load_before_pct": s["host_clock_before"].get("load_pct"), "power": man["host"]["battery_status"],
            "battery_pct": man["host"]["battery_charge_pct"], "commit": man["repo_state"]["commit"], "dirty_code_paths": man["repo_state"]["dirty_code_paths"]})
    out = {"clip": "hold_L05 frames 0-2699 (90 s)", "conditions": "ON BATTERY (discharging 77-80%), Balanced plan, interleaved YT_t8 / S0_t4 x2, WSL2 laptop; "
           "host clock sampled before/after; frequency/thermal unobserved; NOT comparable with the EXP-0010B AC-power timing; energy NOT measured", "configs": rows}
    (RES / "timing_summary.json").write_text(json.dumps(out, indent=1))
    for k, v in rows.items():
        print(k, "p50", [round(x["p50"], 1) for x in v], "p99", [round(x["p99"], 1) for x in v], "drop%", [round(100 * x["dropped_fraction"], 2) for x in v], v[0]["power"])


if __name__ == "__main__":
    main()
