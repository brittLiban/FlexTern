// =============================================================================
// y_stream_engine_tb.sv — testbench for the packed ternary streaming engine
// =============================================================================
//
// Loads one test case written by analysis/hardware_gate/vectors.py:
//   <case>_cfg.txt : "K N n_ops rows acc_w reserved_expected"
//   <case>_w.hex   : packed bytes, one per line, row-major [rows, K]
//                    (checkpoint layout: byte (r, c) at line r*K + c)
//   <case>_x<m>.hex, <case>_y<m>.hex : activations and expected outputs of op m
//
// The testbench IS the memory system: it streams packed bytes to the DUT in
// checkpoint order (no repacking anywhere), serves the activation read port,
// collects result groups, un-strides them (row = i*(N/4) + rowq + p) and
// compares with the Python golden reference.  Operations are issued back to
// back (op m+1 starts as soon as busy drops, while op m's pipeline drains).
//
// +case=<prefix>   required
// +mode=0          continuous: w_valid always 1, y_ready always 1; checks II=1:
//                  last result cycle == first accept cycle + chunks + 1
// +mode=1          random w_valid (50%) and y_ready (50%); results must be
//                  identical; checks w_ready drops only while a result waits
// +out=<file>      dump y per row (for cross-engine comparison)
// +seed=<n>
//
// Compile with -P y_stream_engine_tb.BYTES=.. -P y_stream_engine_tb.ROWQ=..
// Add -DINT4_DUT to test int4_stream_baseline instead (2 rows per byte).
//
// Success line:  PASS: y_stream_engine verification completed.
// =============================================================================

`timescale 1ns/1ps

module y_stream_engine_tb;

    parameter int BYTES = 64;
    parameter int ROWQ  = 1;
`ifdef INT4_DUT
    parameter int ACC_W = 25;
    localparam int RPB  = 2;                       // rows per byte
`else
    parameter int ACC_W = 22;
    localparam int RPB  = 4;
`endif
    localparam int NACC = RPB * ROWQ;
    localparam int MAXW = 1 << 21;                 // packed bytes
    localparam int MAXK = 16384;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                     op_start;
    logic [15:0]              k_len, n_rowq;
    logic                     busy;
    logic                     w_valid, w_ready;
    logic [8*BYTES*ROWQ-1:0]  w_data;
    logic [15:0]              x_col;
    logic [8*BYTES-1:0]       x_data;
    logic                     y_valid, y_ready;
    logic [NACC*ACC_W-1:0]    y_data;
    logic [15:0]              y_rowq;
    logic [ROWQ-1:0]          y_mask;
    logic                     reserved_seen;

`ifdef INT4_DUT
    int4_stream_baseline #(.BYTES(BYTES), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n), .op_start(op_start), .k_len(k_len), .n_rowq(n_rowq), .busy(busy),
        .w_valid(w_valid), .w_ready(w_ready), .w_data(w_data), .x_col(x_col), .x_data(x_data),
        .y_valid(y_valid), .y_ready(y_ready), .y_data(y_data), .y_rowq(y_rowq), .y_mask(y_mask));
    assign reserved_seen = 1'b0;
`else
    y_stream_engine #(.BYTES(BYTES), .ROWQ(ROWQ), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n), .op_start(op_start), .k_len(k_len), .n_rowq(n_rowq), .busy(busy),
        .w_valid(w_valid), .w_ready(w_ready), .w_data(w_data), .x_col(x_col), .x_data(x_data),
        .y_valid(y_valid), .y_ready(y_ready), .y_data(y_data), .y_rowq(y_rowq), .y_mask(y_mask),
        .reserved_seen(reserved_seen));
`endif

    // ------------------------------------------------------------ memories
    logic [7:0]       w_mem [0:MAXW-1];
    logic [7:0]       x_mem [0:MAXK-1];
    logic [ACC_W-1:0] y_exp [0:MAXK-1];
    logic [ACC_W-1:0] y_got [0:MAXK-1];
    logic             y_seen[0:MAXK-1];

    int K, N, n_ops, rows, acc_w_file, res_exp, nrowq, mode, seed;
    string case_prefix, out_path, tmp;
    int errors = 0;

    // activation read port (combinational read of the external buffer).  Written as an
    // explicitly-sensitive block (x_col plus a reload counter) rather than always_comb:
    // Icarus makes always_comb sensitive to every element of the array it reads.
    int xver = 0;
    integer jj;
    always @(x_col, xver) begin
        for (jj = 0; jj < BYTES; jj = jj + 1) x_data[8*jj +: 8] = x_mem[x_col + jj];
    end

    // weight stream driver: presents packed bytes in checkpoint order
    int s_rowq, s_col, wver = 0;
    logic s_done;
    integer pp, qq;
    always @(s_rowq, s_col, wver) begin
        for (pp = 0; pp < ROWQ; pp = pp + 1)
            for (qq = 0; qq < BYTES; qq = qq + 1)
                w_data[8*(pp*BYTES + qq) +: 8] = ((s_rowq + pp) < nrowq) ? w_mem[(s_rowq + pp)*K + s_col + qq] : 8'h00;
    end

    // ------------------------------------------------------------ monitors
    longint cycle = 0;
    longint first_accept = -1, last_accept = -1, last_result = -1, n_accepts = 0, n_results = 0;
    longint stall_viol = 0;    // w_ready low while active and no result pending (mode 1 check)
    int groups_expected, groups_got;

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (rst_n) begin
            if (w_valid && w_ready) begin
                n_accepts <= n_accepts + 1;
                if (first_accept < 0) first_accept <= cycle;
                last_accept <= cycle;
            end
            if (busy && !w_ready && !(y_valid && !y_ready)) stall_viol <= stall_viol + 1;
            if (y_valid && y_ready) begin
                n_results <= n_results + 1;
                last_result <= cycle;
                groups_got <= groups_got + 1;
                for (int p = 0; p < ROWQ; p = p + 1) begin
                    if (y_mask[p]) begin
                        for (int i = 0; i < RPB; i = i + 1) begin
                            y_got[i*rows + y_rowq + p]  <= y_data[ACC_W*(RPB*p + i) +: ACC_W];
                            y_seen[i*rows + y_rowq + p] <= 1'b1;
                        end
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------ driver
    task automatic run_op(input int m);
        int fd; string fn; longint t0, chunks;
        begin
            $sformat(fn, "%s_x%0d.hex", case_prefix, m); $readmemh(fn, x_mem, 0, K-1); xver = xver + 1;
            $sformat(fn, "%s_y%0d.hex", case_prefix, m); $readmemh(fn, y_exp, 0, N-1);
            for (int r = 0; r < N; r = r + 1) y_seen[r] = 1'b0;
            groups_got = 0; groups_expected = (nrowq + ROWQ - 1) / ROWQ;
            first_accept = -1; n_accepts = 0;
            // wait for idle, start
            while (busy) @(posedge clk);
            @(negedge clk);
            k_len = K; n_rowq = nrowq; op_start = 1'b1; s_rowq = 0; s_col = 0; s_done = 1'b0;
            t0 = cycle;
            @(negedge clk); op_start = 1'b0;
            // stream
            chunks = 0;
            while (!s_done) begin
                w_valid = (mode == 0) ? 1'b1 : ($urandom % 2);
                y_ready = (mode == 0) ? 1'b1 : ($urandom % 2);
                @(posedge clk);
                if (w_valid && w_ready) begin
                    chunks = chunks + 1;
                    #1;
                    s_col = s_col + BYTES;
                    if (s_col >= K) begin s_col = 0; s_rowq = s_rowq + ROWQ; if (s_rowq >= nrowq) s_done = 1'b1; end
                end
                @(negedge clk);
            end
            w_valid = 1'b0;
            // drain
            while (groups_got < groups_expected) begin
                y_ready = (mode == 0) ? 1'b1 : ($urandom % 2);
                @(negedge clk);
            end
            y_ready = 1'b1;
            @(negedge clk);
            // compare
            for (int r = 0; r < N; r = r + 1) begin
                if (!y_seen[r]) begin errors++; if (errors < 10) $display("ERROR op %0d row %0d never produced", m, r); end
                else if (y_got[r] !== y_exp[r]) begin
                    errors++;
                    if (errors < 10) $display("ERROR op %0d row %0d got %0d exp %0d", m, r, $signed(y_got[r]), $signed(y_exp[r]));
                end
            end
            if (mode == 0) begin
                // II = 1: chunks accepted on consecutive cycles, result 2 cycles after the last accept
                if (n_accepts != chunks) begin errors++; $display("ERROR accept count %0d != chunks %0d", n_accepts, chunks); end
                if (last_result != first_accept + chunks + 1) begin
                    errors++; $display("ERROR II/latency: first_accept %0d chunks %0d last_result %0d (expected %0d)",
                                       first_accept, chunks, last_result, first_accept + chunks + 1);
                end
                $display("op %0d: K=%0d N=%0d chunks=%0d cycles(start->last result)=%0d  II=1 check ok=%0d",
                         m, K, N, chunks, last_result - t0, (last_result == first_accept + chunks + 1));
            end else begin
                $display("op %0d: K=%0d N=%0d chunks=%0d cycles=%0d (random valid/ready) stall_violations=%0d",
                         m, K, N, chunks, last_result - t0, stall_viol);
            end
            // dump
            if (out_path != "") begin
                fd = $fopen(out_path, (m == 0) ? "w" : "a");
                for (int r = 0; r < N; r = r + 1) $fdisplay(fd, "%0d %0d %0d", m, r, $signed(y_got[r]));
                $fclose(fd);
            end
        end
    endtask

    initial begin
        int fd, rc; string fn;
        if (!$value$plusargs("case=%s", case_prefix)) begin $display("FAIL: +case missing"); $finish; end
        if (!$value$plusargs("mode=%d", mode)) mode = 0;
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        if (!$value$plusargs("out=%s", out_path)) out_path = "";
        rc = $urandom(seed);
        $sformat(fn, "%s_cfg.txt", case_prefix);
        fd = $fopen(fn, "r");
        if (fd == 0) begin $display("FAIL: cannot open %s", fn); $finish; end
        rc = $fscanf(fd, "%d %d %d %d %d %d", K, N, n_ops, rows, acc_w_file, res_exp);
        $fclose(fd);
        if (acc_w_file != ACC_W) begin $display("FAIL: case acc_w %0d != ACC_W %0d", acc_w_file, ACC_W); $finish; end
        if (K % BYTES != 0) begin $display("FAIL: K=%0d not a multiple of BYTES=%0d", K, BYTES); $finish; end
        nrowq = rows;
        $sformat(fn, "%s_w.hex", case_prefix); $readmemh(fn, w_mem, 0, rows*K-1);

        op_start = 0; w_valid = 0; y_ready = 1; s_rowq = 0; s_col = 0; s_done = 1; wver = wver + 1;
        repeat (3) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);

        for (int m = 0; m < n_ops; m = m + 1) run_op(m);

        if (reserved_seen !== res_exp[0]) begin errors++; $display("ERROR reserved_seen=%0d expected %0d", reserved_seen, res_exp); end
        if (mode == 1 && stall_viol != 0) begin errors++; $display("ERROR w_ready dropped without a pending result %0d times", stall_viol); end

        if (errors != 0) $display("FAIL: y_stream_engine verification found %0d error(s).", errors);
        else begin
            $display("%0d op(s), N=%0d rows each, all outputs match the Python reference.", n_ops, N);
            $display("PASS: y_stream_engine verification completed.");
        end
        $finish;
    end

endmodule
