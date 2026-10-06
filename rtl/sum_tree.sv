// =============================================================================
// sum_tree.sv — balanced signed adder tree, N_IN inputs of W_IN bits
// =============================================================================
//
// out = sum of N_IN signed W_IN-bit inputs, N_IN a power of two.
// Output width grows by one bit per level: W_IN + log2(N_IN), so no partial
// sum can overflow.  Recursive instantiation gives a balanced binary tree
// (depth log2(N_IN) adders) instead of the sequential chain used in
// y_dot16.sv v0.1.  Purely combinational.
//
// in_flat[W_IN*i +: W_IN] = input i (two's complement).
// =============================================================================

`timescale 1ns/1ps

module sum_tree #(
    parameter int N_IN = 16,
    parameter int W_IN = 9
) (
    input  logic [N_IN*W_IN-1:0]                  in_flat,
    output logic signed [W_IN+$clog2(N_IN)-1:0]   out
);

    generate
        if (N_IN == 1) begin : g_leaf
            assign out = $signed(in_flat[W_IN-1:0]);
        end else begin : g_node
            localparam int HALF = N_IN / 2;
            localparam int WH   = W_IN + $clog2(HALF);
            logic signed [WH-1:0] lo, hi;
            sum_tree #(.N_IN(HALF), .W_IN(W_IN)) u_lo (.in_flat(in_flat[HALF*W_IN-1:0]),          .out(lo));
            sum_tree #(.N_IN(HALF), .W_IN(W_IN)) u_hi (.in_flat(in_flat[N_IN*W_IN-1:HALF*W_IN]), .out(hi));
            assign out = $signed({lo[WH-1], lo}) + $signed({hi[WH-1], hi});
        end
    endgenerate

endmodule
