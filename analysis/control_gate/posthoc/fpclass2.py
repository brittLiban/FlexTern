import json, numpy as np
from analysis.control_gate import run_cag, zones
from analysis.edge_inference import evaluate as ev
names={0:"Person",1:"Forklift",2:"NovaCarter",3:"Transporter",4:"FourierGR1T2",5:"AgilityDigit",6:"PalletTruck"}
res={}
for role in ("hold_L08","hold_L02","dev_L11","dev_L10"):
    clip,_=run_cag.load_clip(role); g=clip.g
    cnt={names[c]:int((g["cls"]==c).sum()) for c in np.unique(g["cls"])}
    cfg="YT"; stride=1 if role.startswith("hold") else 3
    dets,n=run_cag.load_dets(role,cfg,stride)
    pos,neg=zones.gt_states(clip.n_frames,clip.foot,clip.frame,clip.vis_h,clip.depth_ok)
    k=(dets["score"]>=0.5)&((dets["box"][:,3]-dets["box"][:,1])>=32)
    f,b=dets["frame"][k],dets["box"][k]
    foot=np.stack([(b[:,0]+b[:,2])/2,np.minimum(b[:,3],1079)],1); zm=zones.zone_masks(foot,32.0)
    rows_by_f={}
    for i,fr in enumerate(g["frame"]): rows_by_f.setdefault(int(fr),[]).append(i)
    best={}
    tot=0
    for i in range(len(f)):
        if not zm[i].any() or not all(neg[f[i],z] for z in np.nonzero(zm[i])[0]): continue
        tot+=1
        rows=rows_by_f.get(int(f[i]),[])
        if not rows: best["none"]=best.get("none",0)+1; continue
        gb=np.stack([[g["x1"][r],g["y1"][r],g["x2"][r],g["y2"][r]] for r in rows]); iou=ev.box_iou(b[i:i+1],gb)[0]
        j=int(np.argmax(iou)); key=names[int(g["cls"][rows[j]])] if iou[j]>=0.3 else "<0.3 any"
        best[key]=best.get(key,0)+1
    res[role]={"gt_object_counts":cnt,"false_alarm_dets_score>=0.5":tot,"by_best_overlapping_gt_class(iou>=0.3)":best}
    print(role,json.dumps(res[role]),flush=True)
json.dump(res,open(str(run_cag.RESULTS/"posthoc_fa_by_class.json"),"w"),indent=1)
