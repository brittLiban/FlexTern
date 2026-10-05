// =============================================================================
// y_core_tb.sv — exhaustive testbench for Y-Core v0.1
// =============================================================================
//
// Covers the ENTIRE valid input domain:
//   256 activations (-128..+127) x 3 valid weights (+1, 0, -1) = 768 cases.
// Then separately checks the reserved weight encoding 2'b10 for all 256
// activations, verifying the documented behaviour (result = 0).
//
// Expected values are computed in the testbench with 32-bit integer math so
// the DUT's 9-bit result is checked against exact arithmetic, not against
// another 9-bit implementation.
//
// Compile:  iverilog -g2012 -o build/y_core_tb.vvp rtl/y_core.sv tests/y_core_tb.sv
// Run:      vvp build/y_core_tb.vvp
//
// On success the final line is exactly:
//   PASS: Y-Core exhaustive verification completed.
// Any mismatch prints the case and the simulation exits with a fatal error.
// =============================================================================

`timescale 1ns/1ps

module y_core_tb;

    logic signed [7:0] activation;
    logic        [1:0] weight;
    logic signed [8:0] result;

    y_core dut (
        .activation (activation),
        .weight     (weight),
        .result     (result)
    );

    int errors        = 0;
    int valid_cases   = 0;
    int reserved_cases = 0;

    // Map the valid ternary weight value to its 2-bit encoding.
    function automatic logic [1:0] encode(input int w);
        case (w)
             0: encode = 2'b00;
             1: encode = 2'b01;
            -1: encode = 2'b11;
            default: begin
                $fatal(1, "encode(): invalid ternary weight %0d", w);
            end
        endcase
    endfunction

    task automatic check(input int a, input int w, input logic [1:0] enc, input int expected);
        int actual;
        begin
            activation = a[7:0];
            weight     = enc;
            #1;
            actual = result;  // signed 9-bit -> int, sign-extends
            if (actual !== expected) begin
                errors++;
                $display("MISMATCH: activation=%0d weight_enc=%b (w=%0d) expected=%0d actual=%0d",
                         a, enc, w, expected, actual);
            end
        end
    endtask

    initial begin
        int a, w;
        int expected;

        // --- 1. Exhaustive valid domain: 256 x 3 = 768 -----------------------
        // Loop variable is a 32-bit int, so -128..127 inclusive does not wrap.
        for (a = -128; a <= 127; a++) begin
            for (w = -1; w <= 1; w++) begin
                expected = a * w;  // exact integer math in the TB
                check(a, w, encode(w), expected);
                valid_cases++;
            end
        end

        // --- 2. The critical corner case, named explicitly --------------------
        // -128 * -1 = +128 must be representable; this is why result is 9 bits.
        activation = -8'sd128;  // 8'b1000_0000
        weight     = 2'b11;
        #1;
        if (result !== 9'sd128) begin
            errors++;
            $display("MISMATCH (named case): -128 * -1 expected 128, got %0d", result);
        end else begin
            $display("OK: -128 * -1 = +128 (9-bit result verified)");
        end

        // --- 3. Reserved encoding 2'b10 -> documented behaviour: 0 -----------
        for (a = -128; a <= 127; a++) begin
            activation = a[7:0];
            weight     = 2'b10;
            #1;
            if (result !== 9'sd0) begin
                errors++;
                $display("MISMATCH (reserved): activation=%0d weight_enc=10 expected=0 actual=%0d",
                         a, result);
            end
            reserved_cases++;
        end

        // --- Summary ------------------------------------------------------------
        if (valid_cases != 768) begin
            errors++;
            $display("ERROR: expected 768 valid cases, ran %0d", valid_cases);
        end

        if (errors != 0) begin
            $display("FAIL: Y-Core verification found %0d mismatch(es).", errors);
            $fatal(1, "Y-Core verification failed");
        end else begin
            $display("%0d valid ternary cases passed.", valid_cases);
            $display("%0d reserved-encoding (2'b10 -> 0) cases passed.", reserved_cases);
            $display("PASS: Y-Core exhaustive verification completed.");
            $finish;
        end
    end

endmodule
