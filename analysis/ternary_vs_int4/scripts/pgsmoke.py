import json
from analysis.ternary_vs_int4 import perf, config as C, summarize
for rt, mid in (("bitnet", "BITNET_2B4T_I2S"), ("bitnet", "QWEN25_1P5B_Q4KM"), ("upstream", "QWEN25_1P5B_Q4KM")):
    recs = {}
    for kind in ("ttft", "total"):
        r = perf.run_one(mid, rt, 8, 64, 16, kind, 1)
        print(mid, rt, kind, "rc", r["returncode"], "n_prompt", r.get("n_prompt"), "n_gen", r.get("n_gen"), "rss", r.get("max_rss_kb"),
              "samples", r.get("samples_ns"), "power", r["env_before"]["power_check"], r.get("stderr_tail", "")[-300:])
        recs[kind] = r
    m = summarize.metrics(recs["ttft"], recs["total"])
    print("  derived:", {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
