// =============================================================================
// ternary_unpack4.sv — native decode of one checkpoint-packed ternary byte
// =============================================================================
//
// The canonical checkpoints (BitNet 2B4T, Falcon-E 1B/3B; EXP-0002/0002B,
// analysis/safetensors_reader.py) store ternary weights as
//
//     code = q + 1          q in {-1, 0, +1}  ->  code in {0, 1, 2};  3 unused
//
// four codes per byte, packed along the OUTPUT dimension:
//
//     byte at packed[r, c], bits [2i+1:2i] = code of weight W[i*(N/4) + r, c]
//
// i.e. one byte holds the weights of four output rows (r, r+N/4, r+2N/4,
// r+3N/4) at ONE input position c.  This module turns one such byte into four
// Y-Core weight codes (ADR-0002: 00 = 0, 01 = +1, 11 = -1, 10 = reserved -> 0).
//
// The remap is a 2-bit decrement: (code - 1) mod 4 is exactly the two's
// complement of q, which is the Y-Core encoding.
//
//     checkpoint 00 (-1) -> Y-Core 11 (-1)
//     checkpoint 01 ( 0) -> Y-Core 00 ( 0)
//     checkpoint 10 (+1) -> Y-Core 01 (+1)
//     checkpoint 11 (unused) -> Y-Core 10 (reserved) -> contributes 0,
//                               and `reserved` is raised.  DETERMINISTIC and
//                               DOCUMENTED: an unused code never silently
//                               becomes +1 or -1.
//
// No weight is expanded beyond 2 bits; this is wiring plus four 2-bit
// decrementers.  Purely combinational.
// =============================================================================

`timescale 1ns/1ps

module ternary_unpack4 (
    input  logic [7:0] byte_in,    // packed[r, c]
    output logic [7:0] w_flat,     // w_flat[2*i +: 2] = Y-Core code for output row i*(N/4)+r
    output logic       reserved    // any of the four codes was 2'b11
);

    genvar i;
    generate
        for (i = 0; i < 4; i++) begin : gen_code
            assign w_flat[2*i +: 2] = byte_in[2*i +: 2] - 2'd1;
        end
    endgenerate

    assign reserved = (byte_in[1:0] == 2'b11) | (byte_in[3:2] == 2'b11) |
                      (byte_in[5:4] == 2'b11) | (byte_in[7:6] == 2'b11);

endmodule
