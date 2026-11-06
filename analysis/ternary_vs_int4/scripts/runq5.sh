#!/bin/bash
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
L=~/exp0011/quality_run5.log
python -m analysis.ternary_vs_int4.quality --model BITNET_2B4T_I2S --budget-s 100000 >> $L 2>&1
python -m analysis.ternary_vs_int4.quality --model QWEN25_1P5B_Q4KM --runtime bitnet_old --tag crosscheck --budget-s 100000 >> $L 2>&1
echo QUALITY_ALL_DONE >> $L
