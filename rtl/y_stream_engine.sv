// =============================================================================
// y_stream_engine.sv — packed ternary streaming matrix-vector engine (EXP-0005)
// =============================================================================
//
// Computes y[n] = sum_k x[k] * W[n, k] for an N x K ternary matrix stored in
// the canonical checkpoint packing (see ternary_unpack4.sv), consuming the
// packed bytes DIRECTLY: no unpacked weight copy exists anywhere.
//
// Dataflow ("row-quad" native consumption):
//   A packed row r (K consecutive bytes in memory) carries output rows
//   {r, r+N/4, r+2N/4, r+3N/4}.  Each cycle the engine accepts BYTES
//   consecutive bytes of ROWQ packed rows (ROWQ independent row streams), i.e.
//   4*ROWQ*BYTES ternary terms, and adds BYTES terms into each of the
//   4*ROWQ accumulators.  ROWQ = 1 is a single contiguous memory stream (the
//   streaming baseline); ROWQ > 1 needs ROWQ separate address streams
//   (strided by K bytes), which a banked memory can supply.
//
//   Terms per cycle          = 4 * ROWQ * BYTES
//   Packed bytes per cycle   = ROWQ * BYTES
//
// Interface (valid/ready everywhere):
//   op_start/k_len/n_rowq   one operation = N/4 row-quads of K bytes each;
//                           k_len must be a multiple of BYTES (all canonical
//                           K values are multiples of 256).  Sampled when
//                           !busy.  Consecutive operations are supported;
//                           the pipeline of a finished operation drains while
//                           the next one starts.
//   w_valid/w_ready/w_data  packed bytes, w_data[8*(p*BYTES+j) +: 8] =
//                           packed[rowq+p, col+j]
//   x_col/x_data            activation read port of an EXTERNAL buffer
//                           (K x 8 bits, filled before the operation):
//                           x_data[8*j +: 8] = x[x_col + j], same cycle.
//                           Kept outside this module so synthesis reports
//                           logic and memory separately.
//   y_valid/y_ready/y_data  one result group per row-quad group:
//                           y_data[ACC_W*(4*p+i) +: ACC_W] = y[(y_rowq+p) + i*(N/4)]
//                           y_mask[p] = 0 for row streams beyond n_rowq (tail).
//
// Pipeline: stage 0 decode + Y-Core + balanced tree (combinational, one
// cycle), stage 1 registered chunk sums, stage 2 accumulate / emit.  Latency
// from the last accepted chunk to y_valid is 2 cycles.  Initiation interval
// is 1: every cycle with w_valid and w_ready consumes a full chunk.
// w_ready drops only while a finished result waits for y_ready (one output
// register, no FIFO), so a consumer that accepts every result never stalls
// the stream.
//
// Accumulator width ACC_W: 22 bits covers K = 13312 with INT8 activations
// (|sum| <= 128 * 13312 = 1,703,936 < 2^21).  Smaller K need less; see
// results/hardware_gate/ for the per-K widths.
//
// State that exists: 4*ROWQ accumulators, 4*ROWQ chunk-sum registers, one
// output register, counters.  NOTHING proportional to N x K.
// =============================================================================

`timescale 1ns/1ps

module y_stream_engine #(
    parameter int BYTES = 64,   // packed bytes per row stream per cycle (power of two, <= 256)
    parameter int ROWQ  = 1,    // row-quad streams per cycle
    parameter int ACC_W = 22    // signed accumulator width
) (
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       op_start,
    input  logic [15:0]                k_len,
    input  logic [15:0]                n_rowq,
    output logic                       busy,

    input  logic                       w_valid,
    output logic                       w_ready,
    input  logic [8*BYTES*ROWQ-1:0]    w_data,

    output logic [15:0]                x_col,
    input  logic [8*BYTES-1:0]         x_data,

    output logic                       y_valid,
    input  logic                       y_ready,
    output logic [4*ROWQ*ACC_W-1:0]    y_data,
    output logic [15:0]                y_rowq,
    output logic [ROWQ-1:0]            y_mask,

    output logic                       reserved_seen
);

    localparam int NACC = 4 * ROWQ;                 // accumulators
    localparam int TW   = 9 + $clog2(BYTES);        // chunk-sum width (BYTES x 9-bit terms)
    localparam logic [15:0] BYTES16 = BYTES[15:0];
    localparam logic [15:0] ROWQ16  = ROWQ[15:0];

    // ---------------------------------------------------------------- control
    logic        active;
    logic [15:0] col, rowq, k_len_r, n_rowq_r;

    logic en, accept, last_col;
    assign en       = ~y_valid | y_ready;           // pipeline may advance
    assign w_ready  = active & en;
    assign accept   = w_valid & w_ready;
    assign last_col = (col + BYTES16) >= k_len_r;
    assign busy     = active;
    assign x_col    = col;

    // ------------------------------------------------- stage 0: decode + sum
    // Per row stream p and byte j: unpack to four Y-Core codes, one per
    // accumulator i; Y-Core product with x[col+j]; balanced tree over j.
    logic [ROWQ*BYTES-1:0]          res_flags;
    logic [NACC*BYTES*9-1:0]        prod_flat;       // prod_flat[9*((4p+i)*BYTES + j) +: 9]
    logic signed [TW-1:0]           chunk_sum [0:NACC-1];

    genvar p, j, i, a;
    generate
        for (p = 0; p < ROWQ; p++) begin : g_row
            for (j = 0; j < BYTES; j++) begin : g_byte
                logic [7:0] codes;
                ternary_unpack4 u_unpack (
                    .byte_in  (w_data[8*(p*BYTES + j) +: 8]),
                    .w_flat   (codes),
                    .reserved (res_flags[p*BYTES + j])
                );
                for (i = 0; i < 4; i++) begin : g_acc
                    y_core u_core (
                        .activation (x_data[8*j +: 8]),
                        .weight     (codes[2*i +: 2]),
                        .result     (prod_flat[9*((4*p + i)*BYTES + j) +: 9])
                    );
                end
            end
        end
        for (a = 0; a < NACC; a++) begin : g_tree
            sum_tree #(.N_IN(BYTES), .W_IN(9)) u_tree (
                .in_flat (prod_flat[9*a*BYTES +: 9*BYTES]),
                .out     (chunk_sum[a])
            );
        end
    endgenerate

    // ------------------------------------------------- stage 1: chunk register
    logic                 s1_valid, s1_last;
    logic [15:0]          s1_rowq;
    logic [ROWQ-1:0]      s1_mask;
    logic signed [TW-1:0] s1_sum [0:NACC-1];

    // ------------------------------------------------- stage 2: accumulate
    logic signed [ACC_W-1:0] acc [0:NACC-1];

    integer n;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0; col <= '0; rowq <= '0; k_len_r <= '0; n_rowq_r <= '0;
            s1_valid <= 1'b0; s1_last <= 1'b0; s1_rowq <= '0; s1_mask <= '0;
            y_valid <= 1'b0; y_data <= '0; y_rowq <= '0; y_mask <= '0;
            reserved_seen <= 1'b0;
            for (n = 0; n < NACC; n = n + 1) begin acc[n] <= '0; s1_sum[n] <= '0; end
        end else begin
            // -------- operation start (only when idle)
            if (op_start && !active) begin
                k_len_r  <= k_len;
                n_rowq_r <= n_rowq;
                col      <= '0;
                rowq     <= '0;
                active   <= (n_rowq != 16'd0);
            end

            // -------- stage 0 -> 1 (advances when the pipeline is enabled)
            if (en) begin
                s1_valid <= accept;
                s1_last  <= accept & last_col;
                s1_rowq  <= rowq;
                for (n = 0; n < ROWQ; n = n + 1) s1_mask[n] <= ((rowq + n) < n_rowq_r);
                for (n = 0; n < NACC; n = n + 1) s1_sum[n] <= chunk_sum[n];
                if (accept && (|res_flags)) reserved_seen <= 1'b1;
            end

            // -------- counters
            if (accept) begin
                if (last_col) begin
                    col <= '0;
                    rowq <= rowq + ROWQ16;
                    if ((rowq + ROWQ16) >= n_rowq_r) active <= 1'b0;
                end else begin
                    col <= col + BYTES16;
                end
            end

            // -------- output handshake
            if (y_valid && y_ready) y_valid <= 1'b0;

            // -------- stage 1 -> 2
            if (en && s1_valid) begin
                if (s1_last) begin
                    for (n = 0; n < NACC; n = n + 1) begin
                        y_data[ACC_W*n +: ACC_W] <= acc[n] + $signed({{(ACC_W-TW){s1_sum[n][TW-1]}}, s1_sum[n]});
                        acc[n] <= '0;
                    end
                    y_valid <= 1'b1;
                    y_rowq  <= s1_rowq;
                    y_mask  <= s1_mask;
                end else begin
                    for (n = 0; n < NACC; n = n + 1) acc[n] <= acc[n] + $signed({{(ACC_W-TW){s1_sum[n][TW-1]}}, s1_sum[n]});
                end
            end
        end
    end

endmodule
