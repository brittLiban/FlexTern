# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** Phase 2 — Characterize a real ternary transformer workload (BitNet b1.58 2B4T) well enough to determine what hardware problem FlexTern should optimize next.

**Status:** Complete. Awaiting founder/ChatGPT review of EXP-0002. No next implementation objective has been set.

**Owner / agent:** Claude (implementation)

**Started:** 2026-10-06
**Completed:** 2026-10-06

## Why we are doing this

Phase 1 (EXP-0001) showed the ternary primitive is cheap and that accumulation dilutes the advantage. Before designing Y-Array, accumulators, sparsity hardware, or a memory system, FlexTern needs to know what a real ternary transformer actually demands: shapes, dot lengths, zero rates, activation format, scale handling, and where the bytes and arithmetic go.

## Question we are trying to answer

What does a real, publicly released, natively ternary transformer require from the hardware, and which architectural bottleneck should FlexTern attack next?

## Scope

- Select and pin one official native ternary model (ADR-0003).
- Extract configuration programmatically; inventory every linear operation and its shape.
- Measure the -1/0/+1 distribution of every ternary tensor, globally, per module, per layer; measure zero structure along K and dead channels.
- Characterize weight-scale and activation-quantization schemes from the reference implementation; cross-check absmean ternarization against the BF16 latent repo.
- Inventory non-ternary operations per block with static op counts.
- Compute storage sizes, a first-order batch-1 decode traffic model, and mathematically safe accumulator widths for real K.
- Evaluate Y-Core v0.1's assumptions against the model.
- Rank candidate next experiments and recommend exactly one.

## Explicitly out of scope

- Any RTL: Y-Array, accumulator variants, zero-skipping, popcount, packing hardware.
- FPGA bitstreams, board selection, ASIC, compiler, CUDA/Triton.
- Running the model; runtime profiling.
- Performance, power, or area claims derived from model statistics.

## Definition of done

All items met; evidence in `experiments/EXP-0002-real-ternary-workload-profile.md` and `../results/workload_profile/`.

- [x] Canonical docs read first; model selected with provenance (ADR-0003).
- [x] Config extracted; linear shapes inventoried; dot lengths identified.
- [x] Ternary distributions measured; official ternarization reproduced on BF16 latents.
- [x] Zero sparsity measured globally and per layer/module; structure analyzed.
- [x] Scale and activation schemes characterized from code.
- [x] Non-ternary operations inventoried.
- [x] Storage, traffic model, accumulator widths computed.
- [x] Y-Core assumptions evaluated.
- [x] Analysis code validated (16 unit tests + in-tool reconciliation asserts).
- [x] One-command reproduction (`scripts/run_workload_profile.ps1`).
- [x] One next objective recommended, not implemented.

## Required validation

Met. See EXP-0002 "Method" and `tests/test_analysis.py`.

## Relevant files

`analysis/`, `tests/test_analysis.py`, `scripts/run_workload_profile.ps1`, `results/workload_profile/`.

## Relevant ADRs

- ADR-0001 — Operating model.
- ADR-0002 — Y-Core v0.1 interface (the assumptions tested here).
- ADR-0003 — Reference workload: BitNet b1.58 2B4T.

## Relevant experiments

- EXP-0001 — Y-Core v0.1 and Y-Dot16 vs INT8 baseline.
- EXP-0002 — Real ternary workload profile (this objective).

## Known uncertainties

- All findings are from one model. Generalization to other ternary models is Q-018.
- Arithmetic and traffic figures are static estimates, not measurements.

## Questions requiring founder decision

None block this objective. See the EXP-0002 handoff for questions the founder and ChatGPT should discuss before choosing the next objective.
