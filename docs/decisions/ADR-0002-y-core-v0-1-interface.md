# ADR-0002 — Y-Core v0.1 interface, encoding, and measurement baseline

Status: Accepted
Date: 2026-10-05

## Context

The first objective (FT-CURRENT, EXP-0001) could not start until four parameters were fixed: the measurement target, the activation width, the ternary weight encoding, and the definition of the conventional baseline. These were recorded as Q-001 through Q-004 in `OPEN_QUESTIONS.md`. The founder resolved all four by direction on 2026-10-05.

## Decision

1. **Measurement target: generic Yosys synthesis.** No FPGA board, no standard-cell library, no timing constraints. Results are technology-independent cell counts from `synth -flatten`, labeled "generic synthesis / resource comparison," never as area, power, or frequency.
2. **Activation: signed 8-bit** (-128..+127).
3. **Y-Core result: signed 9-bit** (-128..+128), because -128 × -1 = +128 does not fit in 8 bits. Sign extension happens before negation.
4. **Ternary weight encoding: 2-bit**, matching two's complement for the three valid values:
   - `00` = 0
   - `01` = +1
   - `11` = -1
   - `10` = reserved / invalid. Y-Core v0.1 outputs 0 for this input; testbenches verify this.
5. **Y-Dot16 accumulator: signed 13-bit** (16 × ±128 = ±2048; +2048 needs 13 bits).
6. **Conventional baseline: signed INT8 × signed INT8** using the plain `*` operator, 16-bit product, and a 16-element dot-product variant with a signed 20-bit accumulator. It is a generic RTL baseline, not a model of any commercial MAC unit.
7. **Simulator: Icarus Verilog in `-g2012` mode. Synthesizer: Yosys.** RTL is written in the SystemVerilog subset both tools accept: flat packed ports, no unpacked array ports, no interfaces/structs/packages.

## Alternatives Considered

- **Real FPGA target first.** Rejected for v0.1: adds board/toolchain dependency before the primitive is even verified. Can be added later as a separate experiment.
- **1-bit sign + 1-bit zero encoding** (`{zero, sign}`). Equivalent information; the two's-complement style was chosen because it makes the reserved code unambiguous and reads naturally.
- **Fold the reserved code into -1 or +1.** Rejected: silently accepting an invalid encoding hides bugs upstream.
- **INT4 or FP8 baseline.** Deferred. INT8 is the most common conventional low-precision datapath and gives a clear first comparison.

## Why

These are the minimum choices needed to make EXP-0001 reproducible and fair. None is a long-term commitment. The founder chose them as measurement-vehicle parameters, not product decisions.

## Consequences

- Q-001 to Q-004 are closed.
- Any later change to activation width, encoding, or baseline must produce a new ADR and a new experiment; EXP-0001's results stay tied to these parameters.
- Results from EXP-0001 cannot be quoted as FPGA or ASIC numbers.
- The final activation precision (Q-007), final physical weight packing (Q-011), and any physical target remain open.

## Evidence

Founder direction, 2026-10-05 (Phase 1 implementation prompt). No experimental evidence applies; these are setup decisions that make the experiment possible.

## Supersedes

None.

## Superseded By

None.
