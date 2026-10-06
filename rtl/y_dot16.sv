// =============================================================================
// y_dot16.sv — FlexTern Y-Dot16 v0.1
// =============================================================================
//
// Ternary dot product of 16 signed 8-bit activations with 16 ternary weights:
//
//     y = sum_{i=0}^{15}  x[i] * w[i],   w[i] in {-1, 0, +1}
//
// Built from 16 Y-Core instances plus a sequential adder chain. There is no multiplier
// anywhere in this design.
//
// Port packing: ports are flat packed vectors (not unpacked arrays) so the
// module is accepted by both Icarus Verilog and Yosys' SystemVerilog frontend.
//   x_flat[8*i +: 8]  = activation i (signed int8)
//   w_flat[2*i +: 2]  = weight i     (2-bit ternary encoding, see y_core.sv)
//
// Width derivation:
//   each Y-Core result is 9 bits signed, range -128..+128
//   16 terms:  -2048 .. +2048
//   +2048 needs 13 bits signed (range -4096..+4095). 12 bits would hold
//   -2048 but not +2048. Hence 13-bit signed output.
//
// Each 9-bit product is explicitly sign-extended to 13 bits before summation.
// Verilog drops signedness if any operand in an expression is unsigned, so
// relying on implicit extension would be fragile. The extension is spelled
// out with a replication of the sign bit.
//
// Purely combinational. No clock, no pipelining (v0.1 is about correctness
// and structure, not timing).
// =============================================================================

`timescale 1ns/1ps

module y_dot16 (
    input  logic [127:0]      x_flat,  // 16 x signed int8
    input  logic [31:0]       w_flat,  // 16 x 2-bit ternary
    output logic signed [12:0] y        // signed, -2048..+2048
);

    localparam int N = 16;

    // 16 Y-Core results, 9 bits each, packed: prod_flat[9*i +: 9]
    logic [N*9-1:0] prod_flat;

    genvar g;
    generate
        for (g = 0; g < N; g++) begin : gen_cores
            y_core u_core (
                .activation (x_flat[8*g +: 8]),
                .weight     (w_flat[2*g +: 2]),
                .result     (prod_flat[9*g +: 9])
            );
        end
    endgenerate

    // Sum with explicit 13-bit sign extension of every term.
    integer i;
    logic signed [12:0] acc;
    always_comb begin
        acc = 13'sd0;
        for (i = 0; i < N; i = i + 1) begin
            acc = acc + $signed({ {4{prod_flat[9*i + 8]}}, prod_flat[9*i +: 9] });
        end
    end

    assign y = acc;

endmodule
