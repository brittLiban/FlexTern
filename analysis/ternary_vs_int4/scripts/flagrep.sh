#!/bin/bash
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
L=~/exp0011/flag_repeats.log
python -m analysis.ternary_vs_int4.repeat_cell BITNET_2B4T_I2S bitnet_old 12 3840 256 total 2 flag_repeats >> $L 2>&1
python -m analysis.ternary_vs_int4.repeat_cell QWEN25_3B_Q4KM upstream 12 3840 256 total 2 flag_repeats >> $L 2>&1
python -m analysis.ternary_vs_int4.repeat_cell QWEN25_1P5B_Q4KM upstream 12 1024 256 total 2 flag_repeats >> $L 2>&1
echo FLAGREP_DONE >> $L
