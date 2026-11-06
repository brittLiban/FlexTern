#!/bin/bash
# detached, resumable quality run (one process at a time; per-task result files make restarts safe)
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
for i in 1 2 3; do
  python -m analysis.ternary_vs_int4.quality --budget-s 100000 >> ~/exp0011/quality_run2.log 2>&1 && break
  sleep 10
done
echo QUALITY_ALL_DONE >> ~/exp0011/quality_run2.log
