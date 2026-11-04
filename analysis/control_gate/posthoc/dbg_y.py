import numpy as np
from analysis.control_gate import detectors as D, parity as P
from analysis.edge_inference.common import cache_dir
z=np.load(cache_dir()/"cag/dets/dev_L11/YT_s3.npz"); print("dets",len(z["score"]), z["score"].max() if len(z["score"]) else None)
video=cache_dir()/"raw/val/Warehouse_022/videos/Camera_0002.mp4"
rgb=P._frames(video,[300])[300]
eng,r=D.build_engine("YT",4)
out=r(eng.pre(rgb)); print(out.shape, out.dtype)
p=out[0]; print("obj max",p[:,4].max(),"cls0 max",p[:,5].max(),"prod max",(p[:,4]*p[:,5]).max(), "cls max over classes", p[:,5:].max())
print("raw xy range",p[:,:2].min(),p[:,:2].max())
