// =============================================================================
// int8_dot16_baseline.sv — conventional INT8 x INT8 16-element dot product
// =============================================================================
//
// THIS IS NOT PART OF FLEXTERN. It is the measurement baseline for Y-Dot16.
//
//     y = sum_{i=0}^{15} x[i] * w[i],   x[i], w[i] both signed int8
//
// Built from 16 int8_mul_baseline instances plus a sequential adder chain, mirroring the
// structure of y_dot16.sv so the only material difference is the per-element
// operation (general multiply vs ternary select).
//
// Port packing (flat, for Icarus + Yosys compatibility):
//   x_flat[8*i +: 8] = activation i (signed int8)
//   w_flat[8*i +: 8] = weight i     (signed int8)
//
// Width derivation:
//   each product: 16 bits signed, range -16256..+16384
//   16 terms:     -260096 .. +262144
//   +262144 = 2^18 needs 20 bits signed (range -524288..+524287).
//   19 bits signed tops out at +262143, one short. Hence 20-bit accumulator.
//
// Each 16-bit product is explicitly sign-extended to 20 bits before summing,
// for the same reason given in y_dot16.sv.
//
// Purely combinational.
// =============================================================================

`timescale 1ns/1ps

module int8_dot16_baseline (
    input  logic [127:0]       x_flat,  // 16 x signed int8
    input  logic [127:0]       w_flat,  // 16 x signed int8
    output logic signed [19:0] y        // signed, -260096..+262144
);

    localparam int N = 16;

    logic [N*16-1:0] prod_flat;  // prod_flat[16*i +: 16]

    genvar g;
    generate
        for (g = 0; g < N; g++) begin : gen_muls
            int8_mul_baseline u_mul (
                .activation (x_flat[8*g +: 8]),
                .weight     (w_flat[8*g +: 8]),
                .product    (prod_flat[16*g +: 16])
            );
        end
    endgenerate

    integer i;
    logic signed [19:0] acc;
    always_comb begin
        acc = 20'sd0;
        for (i = 0; i < N; i = i + 1) begin
            acc = acc + $signed({ {4{prod_flat[16*i + 15]}}, prod_flat[16*i +: 16] });
        end
    end

    assign y = acc;

endmodule
