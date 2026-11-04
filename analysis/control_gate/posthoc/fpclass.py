import json, numpy as np
from analysis.control_gate import run_cag, zones
from analysis.edge_inference import evaluate as ev
res={}
for role in ("hold_L08","hold_L02"):
    clip,_=run_cag.load_clip(role); g=clip.g
    dets,n=run_cag.load_dets(role,"YT",1)
    pos,neg=zones.gt_states(clip.n_frames,clip.foot,clip.frame,clip.vis_h,clip.depth_ok)
    k=(dets["score"]>=0.5)&((dets["box"][:,3]-dets["box"][:,1])>=32)
    f,b,s=dets["frame"][k],dets["box"][k],dets["score"][k]
    foot=np.stack([(b[:,0]+b[:,2])/2,np.minimum(b[:,3],1079)],1)
    zm=zones.zone_masks(foot,32.0)
    # false-alarm detections: every zone it falls into (grown) is NEG at that frame
    fa=np.array([zm[i].any() and all(neg[f[i],z] for z in np.nonzero(zm[i])[0]) for i in range(len(f))])
    idx=np.nonzero(fa)[0]
    rows_by_f={}
    for i,fr in enumerate(g["frame"]): rows_by_f.setdefault(int(fr),[]).append(i)
    cat={"matches_person_GT_iou>=0.3":0,"matches_person_GT_small(<32px)":0,"overlaps_vehicle_iou>=0.3":0,"overlaps_person_GT_iou0.1-0.3":0,"nothing":0}
    for i in idx:
        rows=rows_by_f.get(int(f[i]),[])
        if not rows: cat["nothing"]+=1; continue
        gb=np.stack([[g["x1"][r],g["y1"][r],g["x2"][r],g["y2"][r]] for r in rows]); cl=np.array([g["cls"][r] for r in rows])
        iou=ev.box_iou(b[i:i+1],gb)[0]
        pm=cl==0; vm=(cl==1)|(cl==6)
        pi=iou[pm].max() if pm.any() else 0; vi=iou[vm].max() if vm.any() else 0
        if pi>=0.3:
            r=np.array(rows)[pm][np.argmax(iou[pm])]
            if (g["y2"][r]-g["y1"][r])<32: cat["matches_person_GT_small(<32px)"]+=1
            else: cat["matches_person_GT_iou>=0.3"]+=1
        elif vi>=0.3: cat["overlaps_vehicle_iou>=0.3"]+=1
        elif pi>=0.1: cat["overlaps_person_GT_iou0.1-0.3"]+=1
        else: cat["nothing"]+=1
    # for 'matches person' case: distance between det foot and that person's projected foot
    res[role]={"confident_dets":int(len(f)),"false_alarm_dets":int(len(idx)),**cat}
    print(role,res[role],flush=True)
json.dump(res,open(str(run_cag.RESULTS/"posthoc_fa_classes.json"),"w"),indent=1)
