"""Aggregate the interleaved timing replays in results/control_gate/runs/hold_L08 into timing_summary.json."""
import glob
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results" / "control_gate"


def main():
    rows = {}
    for p in sorted(glob.glob(str(RES / "runs" / "hold_L08" / "*" / "replay_t*_timing_r*" / "summary.json"))):
        pp = Path(p)
        cfg, th = pp.parents[1].name, pp.parent.name.split("_")[1]
        s = json.loads(pp.read_text())
        lat = s["latency_ms"]
        rows.setdefault(f"{cfg}_{th}", []).append({
            "run": pp.parent.name, "p50": lat["p50"], "p95": lat["p95"], "p99": lat["p99"], "max": lat["max"],
            "dropped_fraction": s["dropped_fraction"], "stage_ms_mean": s["stage_ms_mean"],
            "clock_before_mhz": s["host_clock_before"].get("cur_mhz"), "clock_after_mhz": s["host_clock_after"].get("cur_mhz"),
            "trace_violations": len(s["trace_violations"]), "avg_busy_cores": s["resources"]["avg_busy_cores"]})
    out = {"clip": "hold_L08 frames 0-2699 (90 s)",
           "conditions": "AC power, Balanced plan, runs interleaved S0_t4, YT_t8, S0_t8, YT_t4 x3, WSL2 on a laptop; host clock sampled "
                         "before/after (WSL cannot see frequency; thermal state unobserved); energy NOT measured",
           "configs": rows}
    (RES / "timing_summary.json").write_text(json.dumps(out, indent=1))
    for k, v in rows.items():
        print(k, "p50", [round(x["p50"], 1) for x in v], "p99", [round(x["p99"], 1) for x in v], "max", [round(x["max"]) for x in v],
              "drop%", [round(100 * x["dropped_fraction"], 2) for x in v],
              "stage", {a: round(b, 1) for a, b in v[0]["stage_ms_mean"].items()}, "cores", [round(x["avg_busy_cores"], 1) for x in v],
              "clk", [(x["clock_before_mhz"], x["clock_after_mhz"]) for x in v])


if __name__ == "__main__":
    main()
