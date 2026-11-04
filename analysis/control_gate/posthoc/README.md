# EXP-0010B post-hoc and debugging scripts (preserved unchanged)

These scripts produced `results/control_gate/posthoc_*.json` and `results/control_gate/fp_audit/*unmatched*`, and the debugging checks quoted in EXP-0010B. During EXP-0010B they lived in an untracked scratch directory; they are committed here byte-for-byte (copied from that scratch directory, not edited) so the preserved results are reproducible. They import `analysis.control_gate.*` as it was at commit fcdfbb2 and expect `FLEXTERN_EDGE_CACHE` and the WSL venv (see `analysis/control_gate/README.md`). Several have hard-coded role names (`hold_L08`, `hold_L02`, `dev_L10`, `dev_L11`). They are not part of any frozen protocol.

| script | produced |
|---|---|
| `posthoc.py` | `posthoc_tau_sweep.json` |
| `fpclass.py`, `fpclass2.py` | `posthoc_fa_classes.json`, `posthoc_fa_by_class.json` |
| `humanoid.py` | `posthoc_humanoid_dontcare.json` |
| `tolerant.py`, `tolsweep.py` | `posthoc_tolerant_metric.json`, `posthoc_tolerant_tau_sweep.json` |
| `montage.py` | attempted montage of unmatched confident false alarms (found none; see EXP-0010B R6) |
| `dbg_y.py`, `dbg_y2.py` | the YOLOX input-convention check behind `protocol_v2_amendment1.json` |
| `dbg_ssd.py` | the torchvision `fixed_size` (width, height) check |
