import json, numpy as np
from analysis.control_gate import run_cag, zones, gt3d, consistency as C
sel=json.load(open(str(run_cag.RESULTS/"dev_selection.json")))
out={}
for role in ("hold_L08","hold_L02"):
    clip,video=run_cag.load_clip(role); g=clip.g
    _,_,calp,cam=run_cag.paths(role)
    camera=gt3d.load_camera(calp,cam)
    pos,neg=zones.gt_states(clip.n_frames,clip.foot,clip.frame,clip.vis_h,clip.depth_ok)
    h=(g["cls"]==4)|(g["cls"]==5)
    foot=np.stack([g["cx"][h],g["cy"][h],g["cz"][h]-g["sz"][h]/2],1)
    u,v,z=gt3d.project(foot,camera); hf=np.stack([u,v],1)
    ok=(z>0.1)&(hf[:,0]>=-64)&(hf[:,0]<1984)&(hf[:,1]>=-64)&(hf[:,1]<1144)
    hm=zones.zone_masks(hf,zones.MARGIN)&ok[:,None]
    hum=np.zeros_like(neg); fr=g["frame"][h]
    for zz in range(9): hum[fr[hm[:,zz]],zz]=True
    # humanoid in a zone -> not a clean negative; keep strict POS (a real person there still counts)
    neg2=neg&~hum
    out[role]={"frac_frame_zones_with_humanoid":float(hum.mean()),"neg_cells_before":int(neg.sum()),"neg_cells_after":int(neg2.sum()),"configs":{}}
    for cfg in ("S0","YT"):
        dets,n=run_cag.load_dets(role,cfg,1); fm=np.arange(min(n,clip.n_frames))
        rows=[]
        for tau in (sel["tau"][cfg],0.3,0.5,0.7):
            pred=zones.det_states(clip.n_frames,dets,tau)
            m=zones.occupancy_metrics(zones.occupancy_counts(pos,neg2,pred,fm))
            bt=zones.block_bootstrap_occupancy(pos[fm],neg2[fm],pred[fm])
            rows.append({"tau":tau,"tpr":m["tpr"],"fpr":m["fpr"],"ba":m["balanced_accuracy"],"ba_ci":bt["balanced_accuracy_ci"],"n_neg":m["n_neg"]})
        out[role]["configs"][cfg]=rows
        print(role,cfg,[(r["tau"],round(r["tpr"],3),round(r["fpr"],3),round(r["ba"],3),[round(x,3) for x in r["ba_ci"]]) for r in rows],flush=True)
json.dump(out,open(str(run_cag.RESULTS/"posthoc_humanoid_dontcare.json"),"w"),indent=1,default=float)
