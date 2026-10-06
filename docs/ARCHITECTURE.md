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

## 3. Y-Array and beyond

**Status: Exploratory** (no design work has been done)

```
Y-Dot16
   ↓
Y-Array        many Y-Dots arranged for matrix–vector / matrix–matrix products
   ↓
ternary matrix engine
```

Open design questions (temporal vs spatial accumulation, accumulator width, dataflow, memory interface) are in `OPEN_QUESTIONS.md` Q-009 to Q-011. EXP-0001's finding that accumulation dominates Y-Dot16 makes these questions more important than they looked before.

## 4. Full-stack direction

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

## 5. Measurement baselines (not FlexTern)

**Status: Implemented, for measurement only** — `rtl/int8_mul_baseline.sv`, `rtl/int8_dot16_baseline.sv`.

Signed INT8 × INT8 via `*`, 16-bit product; 16-element dot with 20-bit accumulator. Same adder-chain structure as Y-Dot16 so the element operation is the only material difference. Verified exhaustively (multiplier) and randomly (dot). Not a model of any commercial MAC.

---

## Change log for this document

- 2026-10-06 — Workload-fit note under Y-Core extended to Falcon-E (EXP-0002B): K range 2048–13312, both checkpoint encodings, scale conventions. No design change.
- 2026-10-06 — Added workload-fit note under Y-Core from EXP-0002. No design change.
- 2026-10-05 — Y-Core v0.1 and Y-Dot16 v0.1 marked Implemented after EXP-0001. Baselines section added.
- 2026-10-05 — Initial version. Nothing implemented.
