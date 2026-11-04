import numpy as np, cv2
from analysis.control_gate import run_cag, zones, parity as P
from analysis.edge_inference import evaluate as ev
role="hold_L08"
clip,video=run_cag.load_clip(role); g=clip.g
dets,n=run_cag.load_dets(role,"YT",1)
pos,neg=zones.gt_states(clip.n_frames,clip.foot,clip.frame,clip.vis_h,clip.depth_ok)
k=(dets["score"]>=0.5)&((dets["box"][:,3]-dets["box"][:,1])>=32)
f,b,s=dets["frame"][k],dets["box"][k],dets["score"][k]
foot=np.stack([(b[:,0]+b[:,2])/2,np.minimum(b[:,3],1079)],1); zm=zones.zone_masks(foot,32.0)
rows_by_f={}
for i,fr in enumerate(g["frame"]): rows_by_f.setdefault(int(fr),[]).append(i)
cand=[]
for i in range(len(f)):
    if not zm[i].any() or not all(neg[f[i],z] for z in np.nonzero(zm[i])[0]): continue
    rows=rows_by_f.get(int(f[i]),[])
    iou=ev.box_iou(b[i:i+1],np.stack([[g["x1"][r],g["y1"][r],g["x2"][r],g["y2"][r]] for r in rows]))[0] if rows else np.zeros(1)
    if iou.max()<0.1: cand.append(i)
pick=[cand[j] for j in np.linspace(0,len(cand)-1,24).astype(int)]
fr_need=sorted({int(f[i]) for i in pick}); imgs=P._frames(video,fr_need)
th=[]
for i in pick:
    img=imgs[int(f[i])].copy()
    for r in rows_by_f.get(int(f[i]),[]):
        col=(0,255,0) if g["cls"][r]==0 else (255,140,0)
        cv2.rectangle(img,(int(g["x1"][r]),int(g["y1"][r])),(int(g["x2"][r]),int(g["y2"][r])),col,3)
    x1,y1,x2,y2=[int(v) for v in b[i]]; cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),4)
    cx,cy=(x1+x2)//2,(y1+y2)//2; hf=max(x2-x1,y2-y1)+60
    c=img[max(cy-hf,0):min(cy+hf,1080),max(cx-hf,0):min(cx+hf,1920)]
    c=cv2.resize(c,(280,280)); cv2.putText(c,"f%d s%.2f"%(f[i],s[i]),(4,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2); th.append(c)
m=np.concatenate([np.concatenate(th[r*6:(r+1)*6],1) for r in range(4)],0)
out=run_cag.RESULTS/"fp_audit"; out.mkdir(exist_ok=True)
cv2.imwrite(str(out/"hold_L08_YT_unmatched_confident_fa.jpg"),m[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,80]); print(len(cand))
