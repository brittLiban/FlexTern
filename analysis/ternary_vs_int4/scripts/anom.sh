#!/bin/bash
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
python -m analysis.ternary_vs_int4.repeat_cell QWEN25_1P5B_Q4KM bitnet_old 8 2048 128 ttft 3
