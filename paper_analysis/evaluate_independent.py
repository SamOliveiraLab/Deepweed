import os, sys, glob
import numpy as np, cv2, torch
from skimage import measure
PA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PA)
import unet_model_class
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = unet_model_class.UNet(n_channels=3, n_classes=3).to(dev)
ck = torch.load(os.path.join(PA,"data_model","model","best_instance_unet.pt"), map_location=dev, weights_only=False)
model.load_state_dict(ck["model_state_dict"]); model.eval()

def dice(a,b):
    s=a.sum()+b.sum(); return 1.0 if s==0 else 2.0*np.logical_and(a,b).sum()/s
def iou(a,b):
    u=np.logical_or(a,b).sum(); return 1.0 if u==0 else np.logical_and(a,b).sum()/u

# dedupe Roboflow augmentation copies: (dims, gt frond count) identifies a source
seen, rows = set(), []
for mf in sorted(glob.glob("/tmp/indep/masks/*.png")):
    stem=os.path.basename(mf)[:-4]
    imf=f"/tmp/indep/images/{stem}.jpg"
    gt=cv2.imread(mf, cv2.IMREAD_GRAYSCALE)>0
    inst=cv2.imread(f"/tmp/indep/inst/{stem}.png", cv2.IMREAD_UNCHANGED)
    rgb=cv2.cvtColor(cv2.imread(imf), cv2.COLOR_BGR2RGB); H,W=rgb.shape[:2]
    key=(W,H,int(inst.max()))
    dup = key in seen
    seen.add(key)
    x=cv2.resize(rgb,(256,256)).astype(np.float32)/255.0; x=(x-0.5)/0.5
    t=torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(dev)
    with torch.no_grad(): pr=model(t).argmax(dim=1).squeeze().cpu().numpy()
    pf=cv2.resize(pr.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
    body, fg = pf==1, pf>0
    rows.append(dict(stem=stem, dup=dup, wh=f"{W}x{H}", n_gt=int(inst.max()),
                     n_pr=int(measure.label(body).max()),
                     body_iou=iou(gt,body), body_dice=dice(gt,body),
                     fg_iou=iou(gt,fg), fg_dice=dice(gt,fg),
                     gt_cov=100*gt.mean(), pr_cov=100*body.mean()))

print("%-12s %-10s %4s %4s %6s %6s %6s %7s %7s" % ("stem","WxH","GT","pred","bIoU","bDice","fgIoU","GTcov%","PRcov%"))
for r in rows:
    print("%-12s %-10s %4d %4d %6.3f %6.3f %6.3f %6.2f%% %6.2f%%  %s" %
          (r["stem"], r["wh"], r["n_gt"], r["n_pr"], r["body_iou"], r["body_dice"],
           r["fg_iou"], r["gt_cov"], r["pr_cov"], "(aug dup)" if r["dup"] else ""))

uniq=[r for r in rows if not r["dup"]]
for lbl, rs in (("ALL 17 (incl. aug duplicates)", rows), ("UNIQUE sources only", uniq)):
    if not rs: continue
    print("\n=== %s : n=%d ===" % (lbl, len(rs)))
    for k in ("body_iou","body_dice","fg_iou","fg_dice"):
        v=[r[k] for r in rs]; print("   %-10s %.4f +/- %.4f" % (k, np.mean(v), np.std(v)))
    print("   fronds: GT=%d predicted=%d  (recall %.1f%%)" %
          (sum(r["n_gt"] for r in rs), sum(r["n_pr"] for r in rs),
           100.0*sum(r["n_pr"] for r in rs)/max(sum(r["n_gt"] for r in rs),1)))
