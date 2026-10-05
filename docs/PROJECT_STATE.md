# FlexTern Project State

**Last verified:** 2026-10-05 (repository inspection; no code or results exist yet)

**Current phase:** Phase 1 — ternary primitive (see `ROADMAP.md`)

**Current objective:** FT-CURRENT — Build and validate Y-Core v0.1 and establish a controlled synthesis comparison against a conventional multiply-based datapath. Status: defined, not started. See `CURRENT_WORK.md`.

**Current architecture status:** Nothing implemented. Y-Core exists only as a concept (`ARCHITECTURE.md`). No RTL, tests, synthesis flows, or results in the repository.

**What is working:** Nothing yet. Only the documentation system exists.

**What has been proven:** Nothing. No FlexTern measurement has been taken.

**What has NOT been proven:**
- That a ternary datapath is smaller, faster, or lower-power than a conventional multiply-based datapath on any target.
- That any such advantage, if it exists at the primitive level, survives at dot-product, array, or transformer scale.
- That memory traffic, rather than arithmetic, is or is not the dominant cost.
- Any claim about tokens/watt, tokens/dollar, or GPU comparison.

**Current measured results:** None. `BENCHMARKS.md` is empty by design.

**Current blockers:** Several parameters of the first objective are undecided by the founder (target FPGA/toolchain, activation bit width, definition of the conventional baseline, ternary encoding). Listed in `CURRENT_WORK.md` and `OPEN_QUESTIONS.md`.

**Current major uncertainty:** Whether the elementary ternary operation is where the advantage lives at all, or whether memory movement, sparsity, encoding, or scheduling matters more. The project intends to pivot if evidence says so.

**Immediate next decision point:** Founder resolves the open parameters for FT-CURRENT so implementation can begin.
