# FlexTern Architecture

This document describes the architecture as currently understood. Every section is tagged **Implemented**, **Planned**, or **Exploratory**.

For *why* a design choice was made, see `decisions/`. This document says *what*.

---

## 1. Y-Core — the ternary primitive

**Status: Implemented (v0.1)** — `rtl/y_core.sv`, verified exhaustively, synthesized generically (EXP-0001).

The Y-Core is the elementary FlexTern operation: one activation interacting with one ternary weight.

```
activation (signed 8-bit)
     ↓
  Y-Core
     ↓
result (signed 9-bit)

weight 01 (+1)  ->  +activation
weight 00 ( 0)  ->  0
weight 11 (-1)  ->  -activation
weight 10       ->  reserved, outputs 0
```

Implementation: sign-extend to 9 bits, then a 3-way select between 0, +x, and -x. Combinational, no clock. Interface fixed by ADR-0002.

MEASURED (generic Yosys): 2 `$eq` + 1 `$neg` + 1 `$pmux` before mapping; 33 generic cells after (Yosys 0.69), 60 (Yosys 0.52). No multiplier inferred.

Workload fit (EXP-0002 and EXP-0002B, reference set ADR-0004): the INT8 per-token activation and ternary weight match both BitNet 2B4T and Falcon-E 1B/3B exactly. Neither family's activation quantizer can emit -128, so the 9-bit result's +128 case never occurs on these workloads; the width is still correct for the format. Real dot lengths are K ∈ {2048, 2560, 6912, 9216, 13312}, all multiples of 256, never 16; per-tensor scaling happens once per output after accumulation in both families and does not touch the primitive. Checkpoint weight codes are 00/01/10 → -1/0/+1 in both families (transformers packing), while Y-Core v0.1 uses 00/01/11; a remap at the load boundary is required either way, and the two families also store the scale with opposite conventions (β vs 1/β), so a FlexTern-internal canonical representation (code, scale) is a loader concern, not a primitive concern.

## 2. Y-Dot16 — first composition

**Status: Implemented (v0.1)** — `rtl/y_dot16.sv`, verified against testbench arithmetic and Python-generated vectors, synthesized generically (EXP-0001).

```
x[0..15] (16 × int8)     w[0..15] (16 × 2-bit ternary)
        ↓                        ↓
   16 × Y-Core  ─────────────────┘
        ↓  (16 × 9-bit)
   sign-extend to 13 bits, sequential adder chain
        ↓
   y (signed 13-bit, -2048..+2048)
```

Flat packed ports (`x_flat[127:0]`, `w_flat[31:0]`) so Icarus and Yosys both accept it. Combinational.

MEASURED: 1,330 generic cells (Yosys 0.69). INFERRED: roughly 60% of that is the 13-bit accumulation, not the Y-Cores. The accumulator is a plain `for`-loop chain; no attempt at carry-save or tree balancing was made in v0.1.

## 3. Packed ternary consumers (EXP-0005)

**Status: Implemented (experimental, v0.1)** — `rtl/ternary_unpack4.sv`, `rtl/sum_tree.sv`, `rtl/y_stream_engine.sv`, `rtl/y_memory_bank.sv`, `rtl/y_near_memory_engine.sv`; verified against a Python reference on synthetic vectors and real checkpoint slices; synthesized generically. These are experiment vehicles, not a frozen Youssouf Architecture (no ADR).

The checkpoint packing (`packed[r, c]` holds output rows r, r+N/4, r+2N/4, r+3N/4 at input position c; code = q+1) is consumed directly. `ternary_unpack4` turns one byte into four Y-Core codes with a 2-bit decrement (checkpoint code − 1 is exactly the ADR-0002 encoding; the unused code 11 maps to Y-Core's reserved code and contributes 0).

```
packed bytes (ROWQ row streams × BYTES bytes / cycle)      x[col .. col+BYTES-1]  (external buffer, K × 8 bits)
          │                                                       │
   ternary_unpack4 × ROWQ·BYTES  ──►  y_core × 4·ROWQ·BYTES  ◄────┘
          │  (4 output rows per byte)
   sum_tree (balanced, BYTES inputs) × 4·ROWQ      stage 0, combinational
          │
   chunk-sum registers                             stage 1
          │
   4·ROWQ accumulators (ACC_W = 22 for K ≤ 13312)  stage 2  ──►  result group per row-quad (valid/ready)
```

`y_stream_engine`: valid/ready on both sides, runtime K and N, initiation interval 1, latency 2, state = 4·ROWQ accumulators plus chunk and output registers only (no N × K state). `y_memory_bank`: a `y_stream_engine` (ROWQ = 1) behind an address generator and a 4-entry FIFO reading a bank-local packed-weight memory (external port, synchronous) whose rows it owns for all K, so it never exchanges partial sums. `y_near_memory_engine`: NB banks (power of two) with contiguous row-quad ownership, per-bank activation ports, fixed-priority result collector. Weight memories and activation buffers are ports, so generic synthesis reports logic only; capacity is accounted analytically.

MEASURED (EXP-0005): see `../results/hardware_gate/summary.md`. `rtl/int4_stream_baseline.sv` is the controlled INT4-weight structural baseline (two rows per byte, `*` products, 25-bit accumulators); not FlexTern.

## 4. Y-Array and beyond

**Status: Exploratory** (EXP-0005 built the first two dataflows; nothing is frozen)

Memory organization for the bank-local form (EXP-0006, ANALYTICAL/INFERRED, no decision): the public evidence supports tens of banks of several MB each on one SRAM-heavy die for 1B–3B ternary models (404–648 mm² for Falcon-E 1B at 0.68–1.09 MB/mm², inside the Graphcore 900 MB / 823 mm² envelope), with activation delivery ≤ 12% of weight bytes up to 128 banks and the EXP-0005 collector bounding useful bank counts. Larger models need SRAM chiplets or DRAM-class capacity; compute inside DRAM stacks does not change the weight-read energy class. No memory technology is chosen (Q-023). Distributed residency across pipeline stages (EXP-0007, ANALYTICAL) is feasible at 7B–400B (40–126 stages, < 0.1% interconnect bytes) but its weight reuse is the microbatch, which loses to a tensor-parallel HBM node from batch ~4; the architecture's datacenter form is not supported beyond small-batch interactive serving. Energy of the bank-local form (EXP-0008, ANALYTICAL on T1 sources, no decision): a bank read at a modern node costs ~0.33–0.62 pJ/bit of which the macro is only 0.03–0.15 and the wire from macro to consumer 0.25–0.32, so bank-to-engine distance matters more than the macro; the resident megabytes leak 0.05–1.0 mW/MB (unmeasured at modern nodes), which is the term that decides whether residency pays at edge duty cycles. Retention organization (EXP-0009, ANALYTICAL, no decision): if the bank macros sit on per-layer retention rails, a rolling two-layer active window with every other layer at a lowered array voltage is the planning policy; it needs layer-granular domains (four-layer domains already lose when hot), its wake (1.8 µs per layer at planning) hides behind layer execution at any edge token rate, and its transition energy (1.3 µJ/token for Falcon-E 1B when every layer wakes every token, after the EXP-0009A correction of a 1000× unit error; EXP-0009 had stated 0.53 mJ) is immaterial against the 1.4 mJ read; for idle gaps the backing store decides (EXP-0009A): a DRAM-backed shutdown never beats the streaming baseline at idle, a flash-backed one beats retention beyond ~29 s cool / ~1.8 s hot; whether the foundry offers such a mode, how its retention rail is supplied (a linear derivation halves the saving) and what the array leaks at operating temperature are open.

```
Y-Dot16
   ↓
Y-Array        many Y-Dots arranged for matrix–vector / matrix–matrix products
   ↓
ternary matrix engine
```

Open design questions (temporal vs spatial accumulation, accumulator width, dataflow, memory interface) are in `OPEN_QUESTIONS.md` Q-009 to Q-011. EXP-0001's finding that accumulation dominates Y-Dot16 makes these questions more important than they looked before.

## 5. Full-stack direction

**Status: Exploratory** (direction, not design)

```
model / framework
      ↓
FlexTern compiler / backend
      ↓
FlexTern runtime
      ↓
Youssouf Architecture        (the hardware architecture; Y-Core is its first primitive)
      ↓
FlexTern accelerator silicon
      ↓
systems / cloud
```

No layer above Y-Dot16 has been designed. This stack may be reordered, trimmed, or abandoned as evidence arrives.

## 6. Measurement baselines (not FlexTern)

**Status: Implemented, for measurement only** — `rtl/int8_mul_baseline.sv`, `rtl/int8_dot16_baseline.sv`.

Signed INT8 × INT8 via `*`, 16-bit product; 16-element dot with 20-bit accumulator. Same adder-chain structure as Y-Dot16 so the element operation is the only material difference. Verified exhaustively (multiplier) and randomly (dot). Not a model of any commercial MAC.

---

## Change log for this document

- 2026-10-08 — §4: retention-organization note corrected per EXP-0009A (wake energy nJ-class, idle policy depends on the backing store). No design change.
- 2026-10-07 — §4: retention-organization note from EXP-0009 (exploratory, no decision).
- 2026-10-07 — §4: bank-read energy and leakage note from EXP-0008 (exploratory, no decision).
- 2026-10-07 — §4: distributed-residency note from EXP-0007 (exploratory, negative result).
- 2026-10-07 — §4: memory-organization note from EXP-0006 (exploratory, no decision).
- 2026-10-07 — Added §3 packed ternary consumers (streaming engine, memory bank, near-memory engine) as Implemented/experimental after EXP-0005; sections renumbered. No ADR; nothing frozen.
- 2026-10-06 — Workload-fit note under Y-Core extended to Falcon-E (EXP-0002B): K range 2048–13312, both checkpoint encodings, scale conventions. No design change.
- 2026-10-06 — Added workload-fit note under Y-Core from EXP-0002. No design change.
- 2026-10-05 — Y-Core v0.1 and Y-Dot16 v0.1 marked Implemented after EXP-0001. Baselines section added.
- 2026-10-05 — Initial version. Nothing implemented.
