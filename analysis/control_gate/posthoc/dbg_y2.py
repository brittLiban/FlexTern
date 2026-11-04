import numpy as np, cv2
from analysis.control_gate import detectors as D, parity as P
from analysis.edge_inference.common import cache_dir
video=cache_dir()/"raw/val/Warehouse_022/videos/Camera_0002.mp4"
frames=P._frames(video,[300,3000,6000])
eng,r=D.build_engine("YT",4)
S=416
def lb(rgb,pad=114.0):
    h,w=rgb.shape[:2]; k=min(S/h,S/w)
    res=cv2.resize(rgb,(int(w*k),int(h*k)),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    o=np.full((S,S,3),pad,np.float32); o[:res.shape[0],:res.shape[1]]=res; return o
def run(x): 
    p=r(np.ascontiguousarray(x.transpose(2,0,1)[None].astype(np.float32)))[0]
    return float((p[:,4]*p[:,5]).max()), int(((p[:,4]*p[:,5])>0.3).sum())
for i,rgb in frames.items():
    a=lb(rgb); 
    print(i,"RGB 0-255",run(a),"BGR 0-255",run(a[:,:,::-1]),"RGB/255",run(a/255.0),"RGB/255 imagenet",run((a/255.0-D.IMAGENET_MEAN)/D.IMAGENET_STD))
