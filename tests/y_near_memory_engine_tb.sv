// =============================================================================
// y_near_memory_engine_tb.sv — testbench for the banked near-memory engine
// =============================================================================
//
// Same case files as y_stream_engine_tb.sv.  The testbench models NB
// bank-local packed-weight memories (synchronous read, one-cycle latency) and
// NB activation buffers.  Bank b's local memory holds global row-quads
// [b*rpb, (b+1)*rpb) with rpb = ceil(rows / NB), stored back to back, so a
// local byte address maps to global line (b*rpb)*K + addr.  Weight preload
// happens before op_start and is not timed.
//
// Results arrive through the top's single collector port with global row-quad
// indices; they are un-strided and compared with the Python reference, and
// optionally dumped with +out= so the Python driver can assert bit-identity
// with the streaming engine's dump.
//
// +case=<prefix>  +mode=0|1 (1 = random y_ready)  +out=<file>  +seed=<n>
// Compile with -P y_near_memory_engine_tb.NB=.. -P y_near_memory_engine_tb.BYTES=..
//
// Success line:  PASS: y_near_memory_engine verification completed.
// =============================================================================

`timescale 1ns/1ps

module y_near_memory_engine_tb;

    parameter int NB    = 4;
    parameter int BYTES = 16;
    parameter int ACC_W = 22;
    localparam int ADDR_W = 24;
    localparam int MAXW = 1 << 21;
    localparam int MAXK = 16384;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                     op_start;
    logic [15:0]              k_len, n_rowq;
    logic                     busy;
    logic [NB-1:0]            wm_rd;
    logic [NB*ADDR_W-1:0]     wm_addr;
    logic [NB*8*BYTES-1:0]    wm_rdata;
    logic [NB*16-1:0]         x_col;
    logic [NB*8*BYTES-1:0]    x_data;
    logic                     y_valid, y_ready;
    logic [4*ACC_W-1:0]       y_data;
    logic [15:0]              y_rowq;
    logic [$clog2(NB)-1:0]    y_bank;
    logic                     reserved_seen;

    y_near_memory_engine #(.NB(NB), .BYTES(BYTES), .ACC_W(ACC_W), .ADDR_W(ADDR_W)) dut (
        .clk(clk), .rst_n(rst_n), .op_start(op_start), .k_len(k_len), .n_rowq(n_rowq), .busy(busy),
        .wm_rd(wm_rd), .wm_addr(wm_addr), .wm_rdata(wm_rdata), .x_col(x_col), .x_data(x_data),
        .y_valid(y_valid), .y_ready(y_ready), .y_data(y_data), .y_rowq(y_rowq), .y_bank(y_bank),
        .reserved_seen(reserved_seen));

    logic [7:0]       w_mem [0:MAXW-1];
    logic [7:0]       x_mem [0:MAXK-1];
    logic [ACC_W-1:0] y_exp [0:MAXK-1];
    logic [ACC_W-1:0] y_got [0:MAXK-1];
    logic             y_seen[0:MAXK-1];

    int K, N, n_ops, rows, acc_w_file, res_exp, mode, seed, rpb;
    string case_prefix, out_path;
    int errors = 0;

    // bank-local activation buffers (NB copies of the same vector); explicit sensitivity,
    // see y_stream_engine_tb.sv
    int xver = 0;
    integer b, j;
    always @(x_col, xver) begin
        for (b = 0; b < NB; b = b + 1)
            for (j = 0; j < BYTES; j = j + 1)
                x_data[8*(b*BYTES + j) +: 8] = x_mem[x_col[16*b +: 16] + j];
    end

    // bank-local weight memories: synchronous read, 1-cycle latency
    longint local_reads [0:NB-1];
    always @(posedge clk) begin
        for (int bb = 0; bb < NB; bb = bb + 1) begin
            if (wm_rd[bb]) begin
                for (int q = 0; q < BYTES; q = q + 1)
                    wm_rdata[8*(bb*BYTES + q) +: 8] <= w_mem[(bb*rpb)*K + wm_addr[ADDR_W*bb +: ADDR_W] + q];
                local_reads[bb] <= local_reads[bb] + 1;
            end
        end
    end

    longint cycle = 0, last_result = -1, n_results = 0;
    int groups_got;
    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (rst_n && y_valid && y_ready) begin
            n_results <= n_results + 1; last_result <= cycle; groups_got <= groups_got + 1;
            for (int i = 0; i < 4; i = i + 1) begin
                y_got[i*rows + y_rowq]  <= y_data[ACC_W*i +: ACC_W];
                y_seen[i*rows + y_rowq] <= 1'b1;
            end
        end
    end

    task automatic run_op(input int m);
        int fd; string fn; longint t0;
        begin
            $sformat(fn, "%s_x%0d.hex", case_prefix, m); $readmemh(fn, x_mem, 0, K-1); xver = xver + 1;
            $sformat(fn, "%s_y%0d.hex", case_prefix, m); $readmemh(fn, y_exp, 0, N-1);
            for (int r = 0; r < N; r = r + 1) y_seen[r] = 1'b0;
            groups_got = 0;
            while (busy) @(posedge clk);
            @(negedge clk);
            k_len = K; n_rowq = rows; op_start = 1'b1; t0 = cycle;
            @(negedge clk); op_start = 1'b0;
            while (groups_got < rows) begin
                y_ready = (mode == 0) ? 1'b1 : ($urandom % 2);
                @(negedge clk);
            end
            y_ready = 1'b1; @(negedge clk);
            for (int r = 0; r < N; r = r + 1) begin
                if (!y_seen[r]) begin errors++; if (errors < 10) $display("ERROR op %0d row %0d never produced", m, r); end
                else if (y_got[r] !== y_exp[r]) begin
                    errors++;
                    if (errors < 10) $display("ERROR op %0d row %0d got %0d exp %0d", m, r, $signed(y_got[r]), $signed(y_exp[r]));
                end
            end
            $display("op %0d: K=%0d N=%0d NB=%0d rpb=%0d cycles(start->last result)=%0d ideal=%0d",
                     m, K, N, NB, rpb, last_result - t0, rpb * (K / BYTES));
            if (out_path != "") begin
                fd = $fopen(out_path, (m == 0) ? "w" : "a");
                for (int r = 0; r < N; r = r + 1) $fdisplay(fd, "%0d %0d %0d", m, r, $signed(y_got[r]));
                $fclose(fd);
            end
        end
    endtask

    initial begin
        int fd, rc; string fn; longint total_reads;
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
        rpb = (rows + NB - 1) / NB;
        $sformat(fn, "%s_w.hex", case_prefix); $readmemh(fn, w_mem, 0, rows*K-1);     // preload, untimed
        for (int bb = 0; bb < NB; bb = bb + 1) local_reads[bb] = 0;

        op_start = 0; y_ready = 1; wm_rdata = '0;
        repeat (3) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);

        for (int m = 0; m < n_ops; m = m + 1) run_op(m);

        total_reads = 0;
        for (int bb = 0; bb < NB; bb = bb + 1) total_reads = total_reads + local_reads[bb];
        if (total_reads * BYTES != longint'(n_ops) * rows * K) begin
            errors++; $display("ERROR local reads x BYTES = %0d, expected %0d", total_reads * BYTES, longint'(n_ops) * rows * K);
        end
        $display("local packed-weight bytes read: %0d (= %0d ops x %0d bytes); global weight bytes during inference: 0",
                 total_reads * BYTES, n_ops, rows * K);
        if (reserved_seen !== res_exp[0]) begin errors++; $display("ERROR reserved_seen=%0d expected %0d", reserved_seen, res_exp); end

        if (errors != 0) $display("FAIL: y_near_memory_engine verification found %0d error(s).", errors);
        else begin
            $display("%0d op(s), N=%0d rows each, all outputs match the Python reference.", n_ops, N);
            $display("PASS: y_near_memory_engine verification completed.");
        end
        $finish;
    end

endmodule
