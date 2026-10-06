// =============================================================================
// y_near_memory_engine.sv — NB bank-local ternary engines + broadcast/collect
// =============================================================================
//
//                    activation x[k]  (one external buffer per bank)
//                           │
//            ┌──────────────┼──────────────┐
//            ▼              ▼              ▼
//        bank 0         bank 1    ...   bank NB-1      (y_memory_bank)
//     rows [0,rpb)   rows [rpb,2rpb)   rows [...]       rpb = ceil(n_rowq / NB)
//            │              │              │
//            └────── fixed-priority result collector ──────► y_*
//
// Same arithmetic, same packing, same activation precision and bit-identical
// results as y_stream_engine; what changes is WHERE the packed bytes are read:
// each bank reads its own memory (local traffic) and no weight byte crosses
// a global path during inference.  Activations are delivered to every bank
// (counted as broadcast traffic in the analytical model); results leave one
// per cycle through the collector (one 4 x ACC_W group per cycle maximum).
//
// Row ownership: contiguous blocks of rpb row-quads; the last banks may own
// fewer or zero rows (tail).  Each bank's output row-quad index is offset by
// its base so the global index is reported.  NB must be a power of two
// (ceil-division by shift; no divider).
//
// Memories (weights, activations) are EXTERNAL per bank; see y_memory_bank.
// =============================================================================

`timescale 1ns/1ps

module y_near_memory_engine #(
    parameter int NB     = 4,
    parameter int BYTES  = 16,
    parameter int ACC_W  = 22,
    parameter int ADDR_W = 24
) (
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       op_start,
    input  logic [15:0]                k_len,
    input  logic [15:0]                n_rowq,      // total row-quads (N/4)
    output logic                       busy,

    output logic [NB-1:0]              wm_rd,
    output logic [NB*ADDR_W-1:0]       wm_addr,
    input  logic [NB*8*BYTES-1:0]      wm_rdata,

    output logic [NB*16-1:0]           x_col,
    input  logic [NB*8*BYTES-1:0]      x_data,

    output logic                       y_valid,
    input  logic                       y_ready,
    output logic [4*ACC_W-1:0]         y_data,
    output logic [15:0]                y_rowq,      // global row-quad index
    output logic [$clog2(NB)-1:0]      y_bank,
    output logic                       reserved_seen
);

    localparam int LOG_NB = $clog2(NB);

    // ------------------------------------------------ row partition (combinational on op_start)
    logic [15:0] rpb;                              // rows per bank
    logic [15:0] base   [0:NB-1];
    logic [15:0] nlocal [0:NB-1];
    logic [15:0] base_r [0:NB-1];                  // registered for output indexing

    assign rpb = (n_rowq + 16'(NB - 1)) >> LOG_NB;

    integer b;
    always_comb begin
        for (b = 0; b < NB; b = b + 1) begin
            base[b]   = (b == 0) ? 16'd0 : base[b-1] + rpb;
            nlocal[b] = (base[b] >= n_rowq) ? 16'd0 :
                        (((n_rowq - base[b]) < rpb) ? (n_rowq - base[b]) : rpb);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (b = 0; b < NB; b = b + 1) base_r[b] <= '0;
        end else if (op_start && !busy) begin
            for (b = 0; b < NB; b = b + 1) base_r[b] <= base[b];
        end
    end

    // ------------------------------------------------ banks
    logic [NB-1:0]            bk_busy, bk_yvalid, bk_yready, bk_res;
    logic [4*ACC_W-1:0]       bk_ydata [0:NB-1];
    logic [15:0]              bk_yrowq [0:NB-1];
    logic [31:0]              bk_reads [0:NB-1];

    genvar g;
    generate
        for (g = 0; g < NB; g++) begin : g_bank
            y_memory_bank #(.BYTES(BYTES), .ACC_W(ACC_W), .ADDR_W(ADDR_W)) u_bank (
                .clk(clk), .rst_n(rst_n),
                .op_start(op_start && !busy), .k_len(k_len), .n_rowq(nlocal[g]), .busy(bk_busy[g]),
                .wm_rd(wm_rd[g]), .wm_addr(wm_addr[ADDR_W*g +: ADDR_W]), .wm_rdata(wm_rdata[8*BYTES*g +: 8*BYTES]),
                .x_col(x_col[16*g +: 16]), .x_data(x_data[8*BYTES*g +: 8*BYTES]),
                .y_valid(bk_yvalid[g]), .y_ready(bk_yready[g]), .y_data(bk_ydata[g]), .y_rowq(bk_yrowq[g]),
                .reserved_seen(bk_res[g]), .local_reads(bk_reads[g])
            );
        end
    endgenerate

    assign busy          = |bk_busy;
    assign reserved_seen = |bk_res;

    // ------------------------------------------------ result collector (fixed priority)
    logic [LOG_NB-1:0] sel;
    logic              any;
    integer s;
    always_comb begin
        any = 1'b0; sel = '0;
        for (s = NB - 1; s >= 0; s = s - 1) begin
            if (bk_yvalid[s]) begin any = 1'b1; sel = s; end
        end
        for (s = 0; s < NB; s = s + 1) bk_yready[s] = any && (sel == s) && y_ready;
    end

    assign y_valid = any;
    assign y_data  = bk_ydata[sel];
    assign y_rowq  = bk_yrowq[sel] + base_r[sel];
    assign y_bank  = sel;

endmodule
