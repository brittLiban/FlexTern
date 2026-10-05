// =============================================================================
// y_core.sv — FlexTern Y-Core v0.1
// =============================================================================
//
// The smallest FlexTern compute element: one signed 8-bit activation
// multiplied by one ternary weight, implemented WITHOUT a multiplier.
//
//   weight encoding (2-bit, matches two's-complement for 0, +1, -1):
//     2'b00 ->  0   result = 0
//     2'b01 -> +1   result = +activation
//     2'b11 -> -1   result = -activation
//     2'b10 -> reserved / invalid. v0.1 produces 0 for this input.
//
// Width note: the result is 9 bits signed, not 8. With an 8-bit activation of
// -128 and weight -1 the mathematically correct answer is +128, which does
// not fit in a signed 8-bit value. The activation is therefore sign-extended
// to 9 bits BEFORE negation so no overflow can occur.
//
// This module is purely combinational. No clock, no reset, no state.
// =============================================================================

`timescale 1ns/1ps

module y_core (
    input  logic signed [7:0] activation,  // signed int8, -128..+127
    input  logic        [1:0] weight,      // ternary, encoded as above
    output logic signed [8:0] result       // signed, -128..+128
);

    // Sign-extend to 9 bits first. {sign, value} is the standard idiom.
    logic signed [8:0] act_ext;
    assign act_ext = {activation[7], activation};

    // Ternary selection. This is a 3-way choice between 0, +x, -x.
    // Negation is the only arithmetic; there is no product term anywhere.
    always_comb begin
        case (weight)
            2'b00:   result = 9'sd0;      //  0
            2'b01:   result = act_ext;    // +1
            2'b11:   result = -act_ext;   // -1 (safe: act_ext already 9 bits)
            default: result = 9'sd0;      // 2'b10 reserved -> 0 (documented)
        endcase
    end

endmodule
