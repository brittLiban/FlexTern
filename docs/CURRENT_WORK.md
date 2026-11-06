# Current Work

There is exactly one active engineering objective at a time. This file is the single active objective. The founder and ChatGPT choose the next objective after review; do not automatically start a backlog item.

---

**No active objective; awaiting founder + ChatGPT review.**

**Last completed objective:** EXP-0011 Stage 1: ternary (BitNet b1.58 2B4T) vs optimized INT4 at matched quality on a CPU (quality, latency, throughput, memory). Completed 2026-11-07. **Verdict: SPEED_LOW** under the locked protocol (headline geomean end-to-end ratio 0.73, 95% bootstrap [0.71, 0.76], tolerance-3 pool {Qwen2.5-3B, SmolLM2-1.7B}; BitNet is faster only than Qwen2.5-3B: decode 1.37x, end-to-end 1.12x, below the 1.3x bar). Records: `experiments/EXP-0011-ternary-vs-int4-stage1.md`, `handoffs/2026-11-07-exp-0011-stage1.md`, `reviews/REV-2026-11-06-exp-0011-stage1-review.md`, `analysis/ternary_vs_int4/`, `results/ternary_vs_int4/` (report.md, attribution.md).

**Not authorized / not started:** Stage 2 physical energy measurement (plan: `experiments/EXP-0011-stage2-energy-plan.md`; needs separate founder approval and would require purchasing equipment), any optimized ternary kernel work, accelerator redesign, EXP-0012. Commercial-validation preparation exists (`reviews/MEMO-2026-11-05-commercial-validation-package.md`); no customer has been contacted.

**Open decisions for the founder (AGENTS section 6):** (1) whether to authorize a bounded investigation of a quantized head and a validated modern or TL2 ternary kernel before any energy purchase; (2) whether Stage 2 is justified (the review says not by Stage 1 with this runtime); (3) the missing second strategic report; (4) commercial discovery thresholds.
