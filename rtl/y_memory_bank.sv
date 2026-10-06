// =============================================================================
// y_memory_bank.sv — one bank of the near-memory engine (EXP-0005)
// =============================================================================
//
// A bank OWNS a contiguous range of packed rows (row-quads) of the ternary
// matrix for ALL K, so it produces complete output sums and never exchanges
// partial sums with other banks.  Its weights live in a bank-local memory
// that is read, never streamed over a global fabric, during inference.
//
//   local packed weight memory (EXTERNAL, synchronous read, 1-cycle latency)
//            │  BYTES bytes / cycle
//            ▼
//   address generator ──► 4-entry FIFO (absorbs read latency and output stalls)
//            │
//            ▼
//   y_stream_engine #(BYTES, ROWQ = 1)  ◄── activation buffer (EXTERNAL,
//            │                               K x 8 bits, bank-local copy)
//            ▼
//   4 result sums per row-quad
//
// The weight memory and the activation buffer are ports, not contents, so
// generic synthesis of this module reports control + datapath logic only;
// their capacity is accounted analytically (results/hardware_gate/).
//
// Address space: local byte address = local_rowq * k_len + col, i.e. the
// bank's rows stored back to back exactly as in the checkpoint.  The row base
// is advanced by k_len at each row end, so no multiplier is needed.
//
// Weight preload is NOT part of this module and NOT part of the timed
// inference: the testbench fills the memory before op_start.
// =============================================================================

`timescale 1ns/1ps

module y_memory_bank #(
    parameter int BYTES  = 16,
    parameter int ACC_W  = 22,
    parameter int ADDR_W = 24
) (
    input  logic                   clk,
    input  logic                   rst_n,

    input  logic                   op_start,
    input  logic [15:0]            k_len,
    input  logic [15:0]            n_rowq,      // row-quads owned by this bank for this operation (0 = idle)
    output logic                   busy,

    // local packed-weight memory, synchronous read
    output logic                   wm_rd,
    output logic [ADDR_W-1:0]      wm_addr,     // byte address of a BYTES-aligned chunk
    input  logic [8*BYTES-1:0]     wm_rdata,    // valid the cycle after wm_rd

    // local activation buffer, combinational read
    output logic [15:0]            x_col,
    input  logic [8*BYTES-1:0]     x_data,

    // results (local row-quad index)
    output logic                   y_valid,
    input  logic                   y_ready,
    output logic [4*ACC_W-1:0]     y_data,
    output logic [15:0]            y_rowq,
    output logic                   reserved_seen,

    // accounting (testbench visibility)
    output logic [31:0]            local_reads  // chunks read from the local memory since reset
);

    localparam int DEPTH = 4;
    localparam int W     = 8 * BYTES;
    localparam logic [15:0]     BYTES16 = BYTES[15:0];
    localparam logic [ADDR_W-1:0] BYTESA = BYTES[ADDR_W-1:0];

    // ------------------------------------------------------- address generator
    logic              ag_active;
    logic [15:0]       ag_col, ag_rowq, k_len_r, n_rowq_r;
    logic [ADDR_W-1:0] ag_rowbase;
    logic              pend;                     // a read was issued last cycle
    logic [2:0]        count;                    // FIFO occupancy
    logic              push, pop, issue, ag_last_col;

    assign ag_last_col = (ag_col + BYTES16) >= k_len_r;
    assign issue   = ag_active && ((count + {2'b00, pend}) < DEPTH[2:0]);
    assign wm_rd   = issue;
    assign wm_addr = ag_rowbase + ag_col;

    // ------------------------------------------------------- FIFO
    logic [W-1:0] fifo [0:DEPTH-1];
    logic [1:0]   wr_ptr, rd_ptr;
    logic         eng_w_valid, eng_w_ready;

    assign push        = pend;
    assign eng_w_valid = (count != 3'd0);
    assign pop         = eng_w_valid & eng_w_ready;

    integer q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ag_active <= 1'b0; ag_col <= '0; ag_rowq <= '0; ag_rowbase <= '0;
            k_len_r <= '0; n_rowq_r <= '0; pend <= 1'b0; count <= '0;
            wr_ptr <= '0; rd_ptr <= '0; local_reads <= '0;
            for (q = 0; q < DEPTH; q = q + 1) fifo[q] <= '0;
        end else begin
            if (op_start && !ag_active) begin
                k_len_r    <= k_len;
                n_rowq_r   <= n_rowq;
                ag_col     <= '0;
                ag_rowq    <= '0;
                ag_rowbase <= '0;
                ag_active  <= (n_rowq != 16'd0);
            end
            pend <= issue;
            if (issue) begin
                local_reads <= local_reads + 32'd1;
                if (ag_last_col) begin
                    ag_col     <= '0;
                    ag_rowbase <= ag_rowbase + {{(ADDR_W-16){1'b0}}, k_len_r};
                    ag_rowq    <= ag_rowq + 16'd1;
                    if ((ag_rowq + 16'd1) >= n_rowq_r) ag_active <= 1'b0;
                end else begin
                    ag_col <= ag_col + BYTES16;
                end
            end
            if (push) begin
                fifo[wr_ptr] <= wm_rdata;
                wr_ptr <= wr_ptr + 2'd1;
            end
            if (pop) rd_ptr <= rd_ptr + 2'd1;
            count <= count + {2'b00, push} - {2'b00, pop};
        end
    end

    // ------------------------------------------------------- engine
    logic eng_busy;
    logic [0:0] mask_unused;

    y_stream_engine #(.BYTES(BYTES), .ROWQ(1), .ACC_W(ACC_W)) u_engine (
        .clk(clk), .rst_n(rst_n),
        .op_start(op_start), .k_len(k_len), .n_rowq(n_rowq), .busy(eng_busy),
        .w_valid(eng_w_valid), .w_ready(eng_w_ready), .w_data(fifo[rd_ptr]),
        .x_col(x_col), .x_data(x_data),
        .y_valid(y_valid), .y_ready(y_ready), .y_data(y_data), .y_rowq(y_rowq), .y_mask(mask_unused),
        .reserved_seen(reserved_seen)
    );

    assign busy = ag_active | eng_busy;

endmodule
