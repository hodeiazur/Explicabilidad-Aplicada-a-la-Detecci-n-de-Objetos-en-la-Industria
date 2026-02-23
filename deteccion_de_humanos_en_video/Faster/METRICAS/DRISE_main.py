# ======================================================
# IMPORTS
# ======================================================
import os
import cv2
import glob
import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sort import Sort
from skimage.segmentation import slic
from scipy.stats import pearsonr
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from drise import DRISEBatch

# ======================================================
# CONFIGURACIÓN
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESH = 0.5
DRISE_SCORE_THRESH = 0.6
HUMAN_CLASS_ID = 1

IMG_DIR = "cam_ta1_ws2/"
GT_JSON = "ground_truth_sinMujerAtras.json"

OUTPUT_DIR = "DRISE_HUMAN_RESULTS_FASTER"
SAL_MAPS_DIR = os.path.join(OUTPUT_DIR, "saliency_maps")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAL_MAPS_DIR, exist_ok=True)

FIXED_SIZE = (640, 640)  # (W, H)

# ======================================================
# MODELO
# ======================================================
def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.3
    return model

model = get_model()
model.load_state_dict(torch.load(
    "modelos_faster_best_fasterrcnn_data_augmentation_transferLearning_SOLO_HUMANO_conReal.pth",
    map_location=DEVICE
))
model.to(DEVICE).eval()

# ======================================================
# SORT
# ======================================================
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# ======================================================
# FUNCIONES AUXILIARES
# ======================================================
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)

def predict(img_rgb):
    t = torch.from_numpy(img_rgb/255.).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model(t)[0]

# ======================================================
# MÉTRICAS
# ======================================================
def deletion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map):
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=100, compactness=10, start_label=0)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = img_rgb.copy()
    c_scores, s_scores = [], []

    for idx in sorted_idx:
        seg_mask = segments == seg_vals[idx]
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = 0
        out = predict(perturbed)

        best_iou, best_score = 0.0, 0.0
        for b,s,l in zip(out["boxes"].cpu().numpy(),
                         out["scores"].cpu().numpy(),
                         out["labels"].cpu().numpy()):
            if l != orig_class_id: continue
            v = iou(orig_box, b)
            if v > best_iou:
                best_iou, best_score = v, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2: return 0.0
    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])
    s = np.array(s_scores[:-1])
    if np.std(v)==0 or np.std(s)==0: return 0.0
    return pearsonr(v, s)[0]

def insertion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map):
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=100, compactness=10, start_label=0)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = cv2.GaussianBlur(img_rgb,(21,21),0)
    c_scores, s_scores = [], []

    for idx in sorted_idx:
        seg_mask = segments == seg_vals[idx]
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = img_rgb[seg_mask]
        out = predict(perturbed)

        best_iou, best_score = 0.0, 0.0
        for b,s,l in zip(out["boxes"].cpu().numpy(),
                         out["scores"].cpu().numpy(),
                         out["labels"].cpu().numpy()):
            if l != orig_class_id: continue
            v = iou(orig_box, b)
            if v > best_iou:
                best_iou, best_score = v, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2: return 0.0
    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    s = np.array(s_scores[1:])
    if np.std(v)==0 or np.std(s)==0: return 0.0
    return pearsonr(v, s)[0]

def sparsity(sal):
    return 1.0 / (sal.mean() + 1e-8)

def compute_pointing_game(sal, gt):
    x1,y1,x2,y2 = map(int,gt)
    y,x = np.unravel_index(np.argmax(sal), sal.shape)
    return int(x1<=x<=x2 and y1<=y<=y2)

def compute_EBPG(sal, gt):
    x1,y1,x2,y2 = map(int,gt)
    S = sal / (sal.sum()+1e-12)
    return float(S[y1:y2+1, x1:x2+1].sum())

def match_prediction_to_gt(pred, gt_boxes, thr=0.5):
    best, best_iou = None, 0
    for gt in gt_boxes:
        v = iou(pred, gt)
        if v > best_iou:
            best, best_iou = gt, v
    return best if best_iou >= thr else None

# ======================================================
# D-RISE
# ======================================================
drise = DRISEBatch(
    model=model,
    input_size=(FIXED_SIZE[1], FIXED_SIZE[0]),
    device=DEVICE,
    N=5000,
    p1=0.25,
    gpu_batch=8
)
drise.generate_masks_rise(N=5000, s=16, p1=0.25, savepath=None)

def drise_saliency(img_rgb, box, cls_id):
    H0,W0 = img_rgb.shape[:2]
    img_small = cv2.resize(img_rgb, FIXED_SIZE)

    tensor = torch.from_numpy(img_small/255.).permute(2,0,1).float().unsqueeze(0).to(DEVICE)

    sx, sy = FIXED_SIZE[0]/W0, FIXED_SIZE[1]/H0
    x1,y1,x2,y2 = box
    bbox = [(x1*sx,y1*sy),(x2*sx,y1*sy),(x2*sx,y2*sy),(x1*sx,y2*sy)]

    sal_small = drise.forward(tensor, [cls_id], bbox)[cls_id]
    sal = cv2.resize(sal_small,(W0,H0))
    sal = (sal - sal.min())/(sal.max()+1e-8)
    return sal

# ======================================================
# CARGAR GT
# ======================================================
with open(GT_JSON) as f:
    coco = json.load(f)

img_id = {i["id"]:i["file_name"] for i in coco["images"]}
gt_boxes = {}
for ann in coco["annotations"]:
    if ann["category_id"]!=HUMAN_CLASS_ID: continue
    name = img_id[ann["image_id"]]
    x,y,w,h = ann["bbox"]
    gt_boxes.setdefault(name,[]).append([x,y,x+w,y+h])

# ======================================================
# LOOP PRINCIPAL
# ======================================================
results=[]
paths = sorted(glob.glob(os.path.join(IMG_DIR,"*.jpg")))

for f_idx,p in enumerate(paths):
    img = cv2.imread(p)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    name = os.path.basename(p)

    out = predict(rgb)
    m = (out["scores"]>=CONF_THRESH)&(out["labels"]==HUMAN_CLASS_ID)
    dets = np.concatenate([out["boxes"][m].cpu().numpy(),
                           out["scores"][m].cpu().numpy()[:,None]],axis=1) if m.any() else np.empty((0,5))
    tracks = tracker.update(dets)

    for i,tr in enumerate(tracks):
        x1,y1,x2,y2,tid = tr.astype(int)
        score = dets[i,4] if i<len(dets) else 1.0

        row = dict(frame=f_idx,track_id=tid,score=score,
                   DC=np.nan,IC=np.nan,Sparsity=np.nan,PG_Hit=np.nan,EBPG=np.nan)

        if score>=DRISE_SCORE_THRESH:
            sal = drise_saliency(rgb,[x1,y1,x2,y2],HUMAN_CLASS_ID)
            row["DC"] = deletion_correlation(rgb,model,[x1,y1,x2,y2],HUMAN_CLASS_ID,score,sal)
            row["IC"] = insertion_correlation(rgb,model,[x1,y1,x2,y2],HUMAN_CLASS_ID,score,sal)
            row["Sparsity"] = sparsity(sal)
            gt = match_prediction_to_gt([x1,y1,x2,y2],gt_boxes.get(name,[]))
            if gt:
                row["PG_Hit"] = compute_pointing_game(sal,gt)
                row["EBPG"] = compute_EBPG(sal,gt)

            plt.imsave(os.path.join(SAL_MAPS_DIR,f"frame{f_idx}_track{tid}.png"),sal,cmap="jet")

        results.append(row)

# ======================================================
# GUARDAR CSV
# ======================================================
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR,"drise_metrics.csv"),index=False)
print("✔ D-RISE + SORT finalizado")
