import json, numpy as np
from analysis.control_gate import run_cag, cag_eval as E, zones
out={}
for role in ("hold_L08","hold_L02"):
    clip,_=run_cag.load_clip(role)
    out[role]={}
    for cfg in ("S0","YT"):
        dets,n=run_cag.load_dets(role,cfg,1)
        frames=np.arange(min(n,clip.n_frames))
        rows=[]
        for t,c in E.zone_counts_only(clip,dets,[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],frames).items():
            m=zones.occupancy_metrics(c); rows.append((t,round(m["tpr"],3),round(m["fpr"],3),round(m["balanced_accuracy"],3)))
        out[role][cfg]=rows
        print(role,cfg,rows,flush=True)
        # margin sensitivity at the frozen tau
json.dump(out,open(str(run_cag.RESULTS/"posthoc_tau_sweep.json"),"w"),indent=1)
