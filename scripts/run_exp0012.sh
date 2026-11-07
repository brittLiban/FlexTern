#!/usr/bin/env bash
# EXP-0012 orchestration (WSL, venv ~/ftvenv). Resumable: every phase skips finished (arm, seed, split, condition, trigger) entries.
# Usage: bash scripts/run_exp0012.sh PHASE   with PHASE in: devgate test adaptive state latency report
source ~/ftvenv/bin/activate
export PS_DATA=$HOME/ps_data PS_CKPT=${PS_CKPT:-$HOME/ps_ckpt} TORCHDYNAMO_DISABLE=1
cd "$(dirname "$0")/.."
R=${PS_RES:-$HOME/ps_res}; mkdir -p "$R"
ALL="R,S4,T16,G0,G,G-Tu,G-Ta,G-I4u,G-I4a"
case "$1" in
  devgate)   # locked G0 gate at 300 episodes per cell on DEV
    for s in 0 1 2; do PS_CONDS=NOM,OCC,DIST,HARD python -m analysis.predictive_state.run_eval dev R,S4,T16,G0,G $s $R/dev_gate_s$s.json & done; wait ;;
  test)      # full-perception test results, once
    for s in 0 1 2; do python -m analysis.predictive_state.run_eval test $ALL $s $R/test_full_s$s.json & done; wait
    python -m analysis.predictive_state.run_eval test G-I8a,G-Ta-s8 0 $R/test_full_extra_s0.json ;;
  adaptive)  # dev-calibrated budgets, TEST conditions NOM/OCC/DIST/HARD
    for s in 0 1 2; do
      PS_CONDS=NOM,OCC,DIST,HARD python -m analysis.predictive_state.run_eval test G,G-Ta,G-Tu,G-I4a,G-I4u $s $R/test_adapt_s$s.json learned@0.33,fixed@0.33,pix@0.33 &
    done; wait
    for s in 0 1 2; do
      PS_CONDS=NOM,OCC,DIST,HARD python -m analysis.predictive_state.run_eval test G,G-Ta $s $R/test_adapt_s$s.json learned@0.5,fixed@0.5,pix@0.5,learned@0.2,fixed@0.2,pix@0.2 &
    done; wait ;;
  state)
    python -m analysis.predictive_state.analyze_state probe G,G0,G-Tu,G-Ta,G-I4u,G-I4a 0,1,2 $R/probe.json
    python -m analysis.predictive_state.analyze_state predict G,G-Tu,G-Ta,G-I4u,G-I4a 0,1,2 $R/predict.json ;;
  latency)   # run alone, laptop on AC (KI-021)
    python -m analysis.predictive_state.latency "$PS_CKPT/G_s0.pt" $R/latency.json "$2" ;;
  report)
    python - <<'PY'
import glob, json, os
R = os.environ.get("PS_RES", os.path.expanduser("~/ps_res"))
merged = {}
for f in sorted(glob.glob(f"{R}/test_full_s*.json") + glob.glob(f"{R}/test_full_extra_s*.json") + glob.glob(f"{R}/test_adapt_s*.json")):
    merged.update(json.load(open(f)))
json.dump(merged, open(f"{R}/test_all.json", "w"))
print("merged", len(merged))
PY
    python -m analysis.predictive_state.report $R/test_all.json $R/probe.json $R/predict.json $R/latency.json results/predictive_state ;;
esac
