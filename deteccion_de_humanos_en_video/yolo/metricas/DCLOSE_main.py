import torch
import cv2
import os
import glob
import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from ultralytics import YOLO
from sort.sort import Sort
from skimage.segmentation import slic
from scipy.stats import pearsonr
import pandas as pd

# ======================
# CONFIGURACIÓN
# ======================
MODEL_PATH = "best.pt"
IMAGES_DIR = "cam_ta1_ws2/"
OUTPUT_DIR = "DCLOSE_SORT_RESULTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAL_MAPS_DIR = os.path.join(OUTPUT_DIR, "saliency_maps")
os.makedirs(SAL_MAPS_DIR, exist_ok=True)
CONF_THRESH = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DCLOSE_SCORE_THRESH = 0.6

YOLO_HUMAN_CLASS_ID = 0
COCO_HUMAN_CATEGORY_ID = 1
GT_JSON = "ground_truth_sinMujerAtras.json"

# ======================
# MODELO YOLO
# ======================
model = YOLO(MODEL_PATH)

# ======================
# TRACKER SORT
# ======================
tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

# ======================
# GROUND TRUTH
# ======================
with open(GT_JSON, "r") as f:
    coco = json.load(f)

image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
gt_boxes_per_image = defaultdict(list)
for ann in coco["annotations"]:
    if ann["category_id"] != COCO_HUMAN_CATEGORY_ID:
        continue
    file_name = image_id_to_name[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    gt_boxes_per_image[file_name].append([x, y, x+w, y+h])

for k in gt_boxes_per_image:
    gt_boxes_per_image[k] = np.array(gt_boxes_per_image[k])

# ======================
# IoU
# ======================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)

# ======================
# Funciones DCLOSE
# ======================
def predict_yolo(img_bgr):
    """Devuelve dict con boxes, scores y labels de YOLO para DCLOSE"""
    results = model.predict(img_bgr, conf=0.0, device=DEVICE, verbose=False)
    boxes, scores, labels = [], [], []
    for r in results:
        boxes.append(r.boxes.xyxy.cpu().numpy())
        scores.append(r.boxes.conf.cpu().numpy())
        labels.append(r.boxes.cls.cpu().numpy().astype(int))
    if boxes:
        boxes = np.concatenate(boxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)
    else:
        boxes = np.empty((0,4))
        scores = np.array([])
        labels = np.array([])
    return {"boxes": boxes, "scores": scores, "labels": labels}

def dclose_map_global_yolo(img_bgr, orig_box, orig_class_id, orig_score,
                           n_levels=[100,200], mask_value=0, iou_thresh=0.3):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H,W,_ = img_rgb.shape
    sal_map = np.zeros((H,W),dtype=np.float32)
    img_float = img_rgb.astype(np.float32)/255.0
    for n_segments in n_levels:
        segments = slic(img_float, n_segments=n_segments, compactness=10, start_label=0)
        for seg_val in np.unique(segments):
            seg_mask = (segments==seg_val)
            perturbed = img_rgb.copy()
            perturbed[seg_mask] = mask_value
            out_pert = predict_yolo(perturbed)
            boxes_p = out_pert["boxes"]
            scores_p = out_pert["scores"]
            labels_p = out_pert["labels"]
            best_iou, best_score = 0.0,0.0
            for b,s,l in zip(boxes_p, scores_p, labels_p):
                if l != orig_class_id: continue
                this_iou = compute_iou(orig_box,b)
                if this_iou > best_iou:
                    best_iou,best_score = this_iou,s
            pred_score_pert = best_score if best_iou>=iou_thresh else 0.0
            delta = orig_score - pred_score_pert
            sal_map[seg_mask] += delta
    sal_map -= sal_map.min()
    sal_map /= (sal_map.max()+1e-8)
    return sal_map

def deletion_correlation_yolo(img_bgr, orig_box, orig_class_id, orig_score, sal_map,
                              mask_value=0, n_levels=[100,200]):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H,W = img_rgb.shape[:2]
    segments = slic(img_rgb.astype(np.float32)/255.0, n_segments=n_levels[0], compactness=10)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]
    perturbed = img_rgb.copy()
    c_scores, s_scores = [], []
    for idx in sorted_idx:
        seg_mask = segments==seg_vals[idx]
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = mask_value
        out = predict_yolo(perturbed)
        best_iou, best_score = 0.0, 0.0
        for b,s,l in zip(out["boxes"], out["scores"], out["labels"]):
            if l != orig_class_id: continue
            iou_val = compute_iou(orig_box,b)
            if iou_val>best_iou: best_iou,best_score=iou_val,s
        c_scores.append(best_score if best_iou>0.3 else 0.0)
    if len(c_scores)<2: return 0.0
    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])
    s = np.array(s_scores[:-1])
    if np.std(v)==0 or np.std(s)==0: return 0.0
    corr,_ = pearsonr(v,s)
    return corr

def insertion_correlation_yolo(img_bgr, orig_box, orig_class_id, orig_score, sal_map,
                               blur_kernel=21, n_levels=[100,200]):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    segments = slic(img_rgb.astype(np.float32)/255.0, n_segments=n_levels[0], compactness=10)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    order = np.argsort(seg_scores)[::-1]
    perturbed = cv2.GaussianBlur(img_bgr,(blur_kernel,blur_kernel),0)
    c_scores, s_scores = [], []
    for idx in order:
        seg_mask = segments==seg_vals[idx]
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = img_bgr[seg_mask]
        out = predict_yolo(perturbed)
        best_iou, best_score = 0.0,0.0
        for b,s,l in zip(out["boxes"], out["scores"], out["labels"]):
            if l!=orig_class_id: continue
            iou_val = compute_iou(orig_box,b)
            if iou_val>best_iou: best_iou,best_score = iou_val,s
        c_scores.append(best_score if best_iou>0.3 else 0.0)
    if len(c_scores)<2: return 0.0
    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    s = np.array(s_scores[1:])
    if np.std(v)==0 or np.std(s)==0: return 0.0
    corr,_ = pearsonr(v,s)
    return corr

def sparsity(sal_map):
    return 1.0 / (sal_map.mean()+1e-8)

def compute_pointing_game(sal_map, gt_box):
    x1,y1,x2,y2 = map(int,gt_box)
    max_pos = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    max_y,max_x = max_pos
    return 1 if (x1<=max_x<=x2 and y1<=max_y<=y2) else 0

def compute_EBPG(sal_map, gt_box):
    x1,y1,x2,y2 = map(int,gt_box)
    S = sal_map.astype(np.float32)
    S_norm = S / (S.sum()+1e-12)
    sub = S_norm[y1:y2+1,x1:x2+1]
    return float(sub.sum())

def match_prediction_to_gt(pred_box, gt_boxes, iou_thresh=0.5):
    best_gt = None
    best_iou_val = 0.0
    for gt in gt_boxes:
        val = compute_iou(pred_box, gt)
        if val > best_iou_val:
            best_iou_val = val
            best_gt = gt
    if best_gt is None or best_iou_val<iou_thresh:
        return None
    return best_gt

# ======================
# LOOP PRINCIPAL
# ======================
image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR,"*.jpg")))
results_list = []

for frame_idx, path in enumerate(image_paths):
    frame = cv2.imread(path)
    frame_name = os.path.basename(path)
    gt_boxes = gt_boxes_per_image.get(frame_name, np.empty((0,4)))

    # YOLO DETECTIONS
    results = model.predict(frame, conf=CONF_THRESH, device=DEVICE, verbose=False)
    dets, scores = [], []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, confs, clss):
            if int(cls)==YOLO_HUMAN_CLASS_ID:
                dets.append(box)
                scores.append(score)
    if len(dets)>0:
        dets = np.hstack([np.array(dets), np.array(scores)[:,None]])
    else:
        dets = np.empty((0,5))

    # SORT
    tracks = tracker.update(dets)
    final_detections = []

    for i, track in enumerate(tracks):
        x1,y1,x2,y2,track_id = track
        box = np.array([x1,y1,x2,y2])
        kalman_tracker = tracker.trackers[i]
        if kalman_tracker.time_since_update==0:
            source="detector"
            score_det = scores[i] if i<len(scores) else 1.0
        else:
            source="kalman"
            score_det = 1.0

        row = {"frame": frame_idx, "track_id": int(track_id),
               "source": source, "score": score_det,
               "DC": np.nan, "IC": np.nan, "Sparsity": np.nan,
               "PG_Hit": np.nan, "EBPG": np.nan,
               "saliency_map_path": None,
               "bbox": box}

        if source=="detector" and score_det>=DCLOSE_SCORE_THRESH:
            sal = dclose_map_global_yolo(frame, box, YOLO_HUMAN_CLASS_ID, score_det)
            row["DC"] = deletion_correlation_yolo(frame, box, YOLO_HUMAN_CLASS_ID, score_det, sal)
            row["IC"] = insertion_correlation_yolo(frame, box, YOLO_HUMAN_CLASS_ID, score_det, sal)
            row["Sparsity"] = sparsity(sal)
            gt_match = match_prediction_to_gt(box, gt_boxes)
            if gt_match is not None:
                row["PG_Hit"] = compute_pointing_game(sal, gt_match)
                row["EBPG"] = compute_EBPG(sal, gt_match)
            sal_path = os.path.join(SAL_MAPS_DIR,f"frame{frame_idx}_track{int(track_id)}.png")
            plt.imsave(sal_path, sal, cmap='jet')
            row["saliency_map_path"] = sal_path

        final_detections.append(row)

    results_list.extend(final_detections)

# ======================
# GUARDAR CSV
# ======================
df = pd.DataFrame(results_list)
csv_path = os.path.join(OUTPUT_DIR,"dclose_yolo_sort_metrics.csv")
df.to_csv(csv_path,index=False)
print(f"CSV guardado en {csv_path}")
print(f"Saliency maps guardados en {SAL_MAPS_DIR}")
