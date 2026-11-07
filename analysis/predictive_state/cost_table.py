"""Write results/predictive_state/cost_table.md: ANALYTICAL per-step MACs and weight/state bytes per arm and encoder fraction."""
import sys
from pathlib import Path

from . import cost_model as C

ARMS = ["R", "S4", "T16", "G", "G-I8a", "G-I4u", "G-I4a", "G-Tu", "G-Ta"]


def main(out):
    md = ["# EXP-0012 analytical cost table (ANALYTICAL, not measured)\n",
          "Per control step, batch 1. Weight bytes: fp=16 bit, int8=8, int4=4 (+2 B scale/out-channel), ternary=2 (+2 B/tensor); first conv, norms and biases not counted as quantized. State read+write at 16 bit (8 bit for -s8). The trigger head (and 8x8 pooling) is charged for recurrent arms in the adaptive rows. Predictor and EMA encoder are excluded (training/analysis only). Ternary MACs are adds.\n",
          "| arm | encoder fraction | trigger charged | MACs (k) | bytes (kB) | encoder share of MACs | encoder share of bytes |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for f, trig in ([(1.0, False)] + ([(0.33, True), (0.5, True)] if a.startswith("G") else [])):
            c = C.step_cost(a, f, trig)
            e = c["enc_frac"] * c["enc_macs"] / c["macs"], c["enc_frac"] * c["enc_bytes"] / c["bytes"]
            md.append(f"| {a} | {f:.2f} | {'yes' if trig else 'no'} | {c['macs']/1e3:.0f} | {c['bytes']/1e3:.1f} | {e[0]:.2f} | {e[1]:.2f} |")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/predictive_state/cost_table.md")
