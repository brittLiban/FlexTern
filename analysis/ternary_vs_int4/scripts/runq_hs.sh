#!/bin/bash
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
python -m analysis.ternary_vs_int4.quality --model BITNET_2B4T_I2S --task hellaswag --budget-s 100000 >> ~/exp0011/quality_run3.log 2>&1
echo QUALITY_ALL_DONE >> ~/exp0011/quality_run3.log
