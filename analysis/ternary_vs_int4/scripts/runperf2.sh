#!/bin/bash
# main grid with automatic restart after power-guard refusals (amendment 7); resumable
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
L=~/exp0011/perf_run.log
for i in $(seq 1 20); do
  python -m analysis.ternary_vs_int4.perf main --budget-s 100000 >> $L 2>&1 && { echo PERF_ALL_DONE >> $L; exit 0; }
  echo "restart $i after refusal $(date)" >> $L
  sleep 60
done
echo PERF_GAVE_UP >> $L
