# FlexTern Architecture

This document describes the architecture as currently understood. Every section is tagged **Implemented**, **Planned**, or **Exploratory**. As of 2026-10-05 nothing is implemented.

For *why* a design choice was made, see `decisions/`. This document says *what*.

---

## 1. Y-Core — the ternary primitive

**Status: Planned** (target of the current objective; no RTL exists)

The Y-Core is the elementary FlexTern operation: one activation interacting with one ternary weight.

```
Activation
     ↓
  Y-Core
     ↓
ternary operation

weight = +1  ->  +activation
weight =  0  ->  zero / skip
weight = -1  ->  -activation
```

Properties the current objective will determine, not assume:

- Activation bit width (undecided; see OPEN_QUESTIONS Q-002)
- Ternary weight encoding (undecided; see Q-003)
- Whether "skip" is a true gated bypass or an arithmetic zero (undecided)
- Resource, timing, and power characteristics versus a conventional multiplier (unmeasured)

## 2. Composition — from core to matrix engine

**Status: Exploratory** (conceptual only; no design work has been done)

The near-term direction is to compose the primitive upward:

```
Y-Core
   ↓
Y-Dot          several Y-Cores feeding an accumulator (a ternary dot product)
   ↓
Y-Array        many Y-Dots arranged to compute a matrix–vector or matrix–matrix product
   ↓
ternary matrix engine
```

Open design questions at this layer (temporal vs spatial accumulation, accumulator width, dataflow, memory interface) are recorded in `OPEN_QUESTIONS.md`. None has been decided.

## 3. Full-stack direction

**Status: Exploratory** (direction, not design)

The long-term aspiration is a full stack optimized around ternary / ultra-low-precision compute:

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

No layer above the Y-Core has been designed. Describing any of them in more detail here would be speculation. This stack may be reordered, trimmed, or abandoned as evidence arrives.

## 4. Conventional baseline

**Status: Planned** (required by the current objective; definition undecided)

Every FlexTern hardware claim must be made against a controlled conventional baseline synthesized on the same target with the same methodology. The current objective requires one. What exactly constitutes a fair "conventional multiply-based datapath" (operand widths, pipelining, whether it is signed, whether it includes accumulation) is not yet decided. See `CURRENT_WORK.md` and Q-004.

---

## Change log for this document

Keep this short. One line per material change, newest first. Detail belongs in ADRs and experiments.

- 2026-10-05 — Initial version. Nothing implemented.
