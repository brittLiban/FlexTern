// =============================================================================
// int8_mul_baseline.sv — conventional signed INT8 x INT8 multiplier
// =============================================================================
//
// THIS IS NOT PART OF FLEXTERN. It exists only as a measurement baseline.
//
// It computes a general signed 8-bit x signed 8-bit product using the plain
// Verilog '*' operator, so the synthesizer infers whatever multiplier
// structure it normally would. The Y-Core is compared against this to
// quantify how much logic ternary specialization removes.
//
// Caveats that MUST accompany any comparison:
//   * This computes a strictly more general function than Y-Core (any INT8
//     weight, not just {-1,0,+1}) and produces a wider output. Part of any
//     logic difference is simply reduced function, not architectural
//     cleverness.
//   * This is a generic RTL baseline, not a model of any commercial
//     accelerator's MAC unit, DSP slice, or tensor core.
//
// Width: 8 x 8 signed -> 16-bit signed product. Worst case -128 * -128 =
// +16384, which fits in 16 bits signed (max +32767). 15 bits would also
// hold it, but 16 is the conventional product width and avoids any doubt.
//
// Purely combinational.
// =============================================================================

`timescale 1ns/1ps

module int8_mul_baseline (
    input  logic signed [7:0]  activation,  // signed int8
    input  logic signed [7:0]  weight,      // signed int8, arbitrary
    output logic signed [15:0] product      // signed int16
);

    assign product = activation * weight;

endmodule
