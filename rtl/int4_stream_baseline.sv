// =============================================================================
// int4_stream_baseline.sv — CONTROLLED STRUCTURAL BASELINE: INT8 x signed INT4
// =============================================================================
//
// THIS IS NOT PART OF FLEXTERN and is NOT a model of any commercial
// accelerator.  It mirrors y_stream_engine (ROWQ = 1) with a 4-bit signed
// weight format packed TWO output rows per byte at one K position:
//
//     byte[r, c] low nibble  = W[r, c]          (two's complement, -8..+7)
//     byte[r, c] high nibble = W[r + N/2, c]
//
// so one byte feeds 2 accumulators instead of 4.  At the same BYTES per cycle
// it performs 2*BYTES terms per cycle (half the ternary engine's 4*BYTES); at
// the same terms per cycle it needs twice the bytes.  Products use the plain
// `*` operator (8 x 4 -> 12 bits signed), as the EXP-0001 baseline did.
//
// Accumulator: |sum| <= 128 * 8 * K = 13,631,488 for K = 13312 < 2^24, so 25
// signed bits (parameter ACC_W).
//
// Same interface and control as y_stream_engine with ROWQ = 1; n_rowq means
// row-PAIRS (N/2) here; y_data[ACC_W*i +: ACC_W] = y[i*(N/2) + y_rowq].
// =============================================================================

`timescale 1ns/1ps

module int4_stream_baseline #(
    parameter int BYTES = 128,
    parameter int ACC_W = 25
) (
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic                     op_start,
    input  logic [15:0]              k_len,
    input  logic [15:0]              n_rowq,
    output logic                     busy,

    input  logic                     w_valid,
    output logic                     w_ready,
    input  logic [8*BYTES-1:0]       w_data,

    output logic [15:0]              x_col,
    input  logic [8*BYTES-1:0]       x_data,

    output logic                     y_valid,
    input  logic                     y_ready,
    output logic [2*ACC_W-1:0]       y_data,
    output logic [15:0]              y_rowq,
    output logic [0:0]               y_mask
);

    localparam int NACC = 2;
    localparam int PW   = 12;
    localparam int TW   = PW + $clog2(BYTES);
    localparam logic [15:0] BYTES16 = BYTES[15:0];

    logic        active;
    logic [15:0] col, rowq, k_len_r, n_rowq_r;
    logic en, accept, last_col;
    assign en       = ~y_valid | y_ready;
    assign w_ready  = active & en;
    assign accept   = w_valid & w_ready;
    assign last_col = (col + BYTES16) >= k_len_r;
    assign busy     = active;
    assign x_col    = col;

    logic [NACC*BYTES*PW-1:0] prod_flat;
    logic signed [TW-1:0]     chunk_sum [0:NACC-1];

    genvar j, i, a;
    generate
        for (j = 0; j < BYTES; j++) begin : g_byte
            for (i = 0; i < 2; i++) begin : g_nib
                logic signed [3:0]  wq;
                logic signed [7:0]  xa;
                logic signed [11:0] pr;
                assign wq = w_data[8*j + 4*i +: 4];
                assign xa = x_data[8*j +: 8];
                assign pr = xa * wq;                       // general multiply (baseline)
                assign prod_flat[PW*(i*BYTES + j) +: PW] = pr;
            end
        end
        for (a = 0; a < NACC; a++) begin : g_tree
            sum_tree #(.N_IN(BYTES), .W_IN(PW)) u_tree (
                .in_flat (prod_flat[PW*a*BYTES +: PW*BYTES]),
                .out     (chunk_sum[a])
            );
        end
    endgenerate

    logic                 s1_valid, s1_last;
    logic [15:0]          s1_rowq;
    logic signed [TW-1:0] s1_sum [0:NACC-1];
    logic signed [ACC_W-1:0] acc [0:NACC-1];

    integer n;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0; col <= '0; rowq <= '0; k_len_r <= '0; n_rowq_r <= '0;
            s1_valid <= 1'b0; s1_last <= 1'b0; s1_rowq <= '0;
            y_valid <= 1'b0; y_data <= '0; y_rowq <= '0; y_mask <= 1'b1;
            for (n = 0; n < NACC; n = n + 1) begin acc[n] <= '0; s1_sum[n] <= '0; end
        end else begin
            if (op_start && !active) begin
                k_len_r <= k_len; n_rowq_r <= n_rowq; col <= '0; rowq <= '0;
                active <= (n_rowq != 16'd0);
            end
            if (en) begin
                s1_valid <= accept;
                s1_last  <= accept & last_col;
                s1_rowq  <= rowq;
                for (n = 0; n < NACC; n = n + 1) s1_sum[n] <= chunk_sum[n];
            end
            if (accept) begin
                if (last_col) begin
                    col <= '0; rowq <= rowq + 16'd1;
                    if ((rowq + 16'd1) >= n_rowq_r) active <= 1'b0;
                end else col <= col + BYTES16;
            end
            if (y_valid && y_ready) y_valid <= 1'b0;
            if (en && s1_valid) begin
                if (s1_last) begin
                    for (n = 0; n < NACC; n = n + 1) begin
                        y_data[ACC_W*n +: ACC_W] <= acc[n] + $signed({{(ACC_W-TW){s1_sum[n][TW-1]}}, s1_sum[n]});
                        acc[n] <= '0;
                    end
                    y_valid <= 1'b1; y_rowq <= s1_rowq; y_mask <= 1'b1;
                end else begin
                    for (n = 0; n < NACC; n = n + 1) acc[n] <= acc[n] + $signed({{(ACC_W-TW){s1_sum[n][TW-1]}}, s1_sum[n]});
                end
            end
        end
    end

endmodule
