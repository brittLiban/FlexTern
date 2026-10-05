// =============================================================================
// int8_baseline_tb.sv — correctness check for the INT8 measurement baselines
// =============================================================================
//
// The baselines are only useful for comparison if they are themselves correct.
//
//   1. int8_mul_baseline: exhaustive, all 256 x 256 = 65,536 signed products.
//   2. int8_dot16_baseline: boundary cases plus N_RANDOM random vectors.
//
// Compile: iverilog -g2012 -o build/int8_baseline_tb.vvp rtl/int8_mul_baseline.sv rtl/int8_dot16_baseline.sv tests/int8_baseline_tb.sv
// Run:     vvp build/int8_baseline_tb.vvp
//
// Success line:  PASS: INT8 baseline verification completed.
// =============================================================================

`timescale 1ns/1ps

module int8_baseline_tb;

    localparam int N        = 16;
    localparam int N_RANDOM = 10000;

    // ---- mul DUT
    logic signed [7:0]  m_a, m_w;
    logic signed [15:0] m_p;
    int8_mul_baseline dut_mul (.activation(m_a), .weight(m_w), .product(m_p));

    // ---- dot DUT
    logic [127:0]       x_flat, w_flat;
    logic signed [19:0] y;
    int8_dot16_baseline dut_dot (.x_flat(x_flat), .w_flat(w_flat), .y(y));

    int errors = 0;
    int mul_cases = 0, det_cases = 0, rand_cases = 0;

    int xs [0:N-1];
    int ws [0:N-1];

    task automatic drive();
        int i;
        begin
            for (i = 0; i < N; i++) begin
                x_flat[8*i +: 8] = xs[i][7:0];
                w_flat[8*i +: 8] = ws[i][7:0];
            end
            #1;
        end
    endtask

    task automatic check_dot(input string label);
        int i, exp, act;
        begin
            drive();
            exp = 0;
            for (i = 0; i < N; i++) exp = exp + xs[i] * ws[i];
            act = y;
            if (act !== exp) begin
                errors++;
                $display("MISMATCH [%s]: expected=%0d actual=%0d", label, exp, act);
            end
        end
    endtask

    task automatic fill(input int xv, input int wv);
        int i;
        for (i = 0; i < N; i++) begin xs[i] = xv; ws[i] = wv; end
    endtask

    initial begin
        int a, w, k, i, exp, act;

        // ===== 1. Exhaustive multiplier ========================================
        for (a = -128; a <= 127; a++) begin
            for (w = -128; w <= 127; w++) begin
                m_a = a[7:0];
                m_w = w[7:0];
                #1;
                exp = a * w;
                act = m_p;
                if (act !== exp) begin
                    errors++;
                    $display("MISMATCH [mul]: %0d * %0d expected=%0d actual=%0d", a, w, exp, act);
                end
                mul_cases++;
            end
        end

        // ===== 2. Dot product boundaries =======================================
        fill(-128, -128); check_dot("all -128 * -128 (expect +262144)"); det_cases++;
        fill(-128,  127); check_dot("all -128 * 127 (expect -260096)");  det_cases++;
        fill( 127,  127); check_dot("all 127 * 127 (expect +258064)");   det_cases++;
        fill( 127, -128); check_dot("all 127 * -128 (expect -260096)");  det_cases++;
        fill(  77,    0); check_dot("all zero weights (expect 0)");       det_cases++;
        fill(   0,  -55); check_dot("all zero activations (expect 0)");   det_cases++;
        for (i = 0; i < N; i++) begin xs[i] = (i%2==0) ? 100 : -100; ws[i] = 3; end
        check_dot("alternating sign (expect 0)"); det_cases++;

        // ===== 3. Random dot products ==========================================
        for (k = 0; k < N_RANDOM; k++) begin
            for (i = 0; i < N; i++) begin
                xs[i] = $signed($urandom_range(0, 255) - 128);
                ws[i] = $signed($urandom_range(0, 255) - 128);
            end
            check_dot($sformatf("random #%0d", k));
            rand_cases++;
        end

        // ===== Summary ==========================================================
        if (mul_cases != 65536) begin
            errors++;
            $display("ERROR: expected 65536 multiplier cases, ran %0d", mul_cases);
        end
        if (errors != 0) begin
            $display("FAIL: INT8 baseline verification found %0d mismatch(es).", errors);
            $fatal(1, "INT8 baseline verification failed");
        end else begin
            $display("%0d exhaustive int8 x int8 multiplier cases passed.", mul_cases);
            $display("%0d deterministic dot16 cases passed.", det_cases);
            $display("%0d randomized dot16 cases passed.", rand_cases);
            $display("PASS: INT8 baseline verification completed.");
            $finish;
        end
    end

endmodule
