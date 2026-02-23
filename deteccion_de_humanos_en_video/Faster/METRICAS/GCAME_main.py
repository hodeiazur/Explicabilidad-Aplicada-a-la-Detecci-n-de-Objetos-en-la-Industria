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
from gcame import GCAME
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from skimage.segmentation import slic
from scipy.stats import pearsonr

# ======================================================
# CONFIGURACIÓN
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESH = 0.5
DCL_SCORE_THRESH = 0.6
HUMAN_CLASS_ID = 1

IMG_DIR = "cam_ta1_ws2/"
GT_JSON = "ground_truth_sinMujerAtras.json"

OUTPUT_DIR = "GCAME_SORT_RESULTS"
SAL_MAPS_DIR = os.path.join(OUTPUT_DIR, "saliency_maps")
os.makedirs(SAL_MAPS_DIR, exist_ok=True)

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
model.load_state_dict(
    torch.load(
        "modelos_faster_best_fasterrcnn_data_augmentation_transferLearning_SOLO_HUMANO_conReal.pth",
        map_location=DEVICE
    )
)
model.to(DEVICE).eval()

# ======================================================
# GCAME
# ======================================================
# target_layers = ["backbone.body.layer4"]
# target_layers = ["roi_heads.box_head.fc7"] #dentro de ROI NO SE PUEDE
#target_layers = ["backbone.fpn.lateral4"] sale todo azul con esto
# target_layers = ["backbone.fpn.fpn_layer4"] sale todo azul con esto
target_layers = ["backbone.body.layer4"]
gcame = GCAME(model, target_layers=target_layers, arch="fasterrcnn")


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
    t = transforms.ToTensor()(img_rgb).to(DEVICE)
    with torch.no_grad():
        return model([t])[0]

def gcame_saliency(img_rgb, box, obj_idx):
    img_tensor = transforms.ToTensor()(img_rgb).to(DEVICE)
    heat = gcame(img_tensor, box, obj_idx=obj_idx)

    if heat is None:
        return None # para el caso de gcame donde ROI sea nula

    sal = cv2.resize(heat, (img_rgb.shape[1], img_rgb.shape[0]))
    sal -= sal.min()
    sal /= (sal.max() + 1e-8)
    return sal

# ======================================================
# MÉTRICAS (MISMAS QUE D-CLOSE)
# ======================================================
def deletion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, n_segments=200):
    H,W = img_rgb.shape[:2]
    segments = slic(img_rgb.astype(np.float32)/255.0, n_segments=n_segments, compactness=10)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    order = np.argsort(seg_scores)[::-1]

    perturbed = img_rgb.copy()
    c_scores, s_scores = [], []

    for idx in order:
        mask = segments == seg_vals[idx]
        s_scores.append(seg_scores[idx])
        perturbed[mask] = 0

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

def insertion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, n_segments=200):
    H,W = img_rgb.shape[:2]
    segments = slic(img_rgb.astype(np.float32)/255.0, n_segments=n_segments, compactness=10)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    order = np.argsort(seg_scores)[::-1]

    perturbed = cv2.GaussianBlur(img_rgb, (21,21), 0)
    c_scores, s_scores = [], []

    for idx in order:
        mask = segments == seg_vals[idx]
        s_scores.append(seg_scores[idx])
        perturbed[mask] = img_rgb[mask]

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

def sparsity(sal_map):
    return 1.0 / (sal_map.mean() + 1e-8)

def compute_pointing_game(sal_map, gt_box):
    x1,y1,x2,y2 = map(int, gt_box)
    y,x = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    return int(x1<=x<=x2 and y1<=y<=y2)

def compute_EBPG(sal_map, gt_box):
    x1,y1,x2,y2 = map(int, gt_box)
    S = sal_map / (sal_map.sum() + 1e-12)
    return float(S[y1:y2+1, x1:x2+1].sum())

def match_prediction_to_gt(pred_box, gt_boxes, thr=0.5):
    best, best_iou = None, 0
    for gt in gt_boxes:
        v = iou(pred_box, gt)
        if v > best_iou:
            best_iou, best = v, gt
    return best if best_iou >= thr else None

# ======================================================
# CARGAR GT
# ======================================================
with open(GT_JSON) as f:
    coco = json.load(f)

image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
gt_boxes_per_image = {}

for ann in coco["annotations"]:
    if ann["category_id"] != HUMAN_CLASS_ID: continue
    img_name = image_id_to_name[ann["image_id"]]
    x,y,w,h = ann["bbox"]
    box = [int(x), int(y), int(x+w), int(y+h)]
    gt_boxes_per_image.setdefault(img_name, []).append(box)

# ======================================================
# LOOP PRINCIPAL
# ======================================================
results = []
image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

for frame_idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_name = os.path.basename(path)
    gt_boxes = gt_boxes_per_image.get(frame_name, [])

    out = predict(img_rgb)
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()

    mask = (scores >= CONF_THRESH) & (labels == HUMAN_CLASS_ID)
    boxes, scores = boxes[mask], scores[mask]
    dets = np.concatenate([boxes, scores[:,None]], axis=1) if len(boxes) else np.empty((0,5))

    tracks = tracker.update(dets)

    for i, trk in enumerate(tracks):
        x1,y1,x2,y2,tid = trk
        box = np.array([x1,y1,x2,y2]).astype(int)
        k = tracker.trackers[i]

        if k.time_since_update == 0:
            source = "detector"
            score = scores[i] if i < len(scores) else 1.0
            obj_idx = i
        else:
            source = "kalman"
            score = 1.0
            obj_idx = None

        row = {
            "frame": frame_idx,
            "track_id": int(tid),
            "source": source,
            "score": score,
            "DC": np.nan,
            "IC": np.nan,
            "Sparsity": np.nan,
            "PG_Hit": np.nan,
            "EBPG": np.nan,
            "saliency_map_path": None
        }

        if source == "detector" and score >= DCL_SCORE_THRESH:
            sal = gcame_saliency(img_rgb, box, obj_idx)
            if sal is None:
                continue 

            row["DC"] = deletion_correlation(img_rgb, model, box, HUMAN_CLASS_ID, score, sal)
            row["IC"] = insertion_correlation(img_rgb, model, box, HUMAN_CLASS_ID, score, sal)
            row["Sparsity"] = sparsity(sal)

            gt = match_prediction_to_gt(box, gt_boxes)
            if gt is not None:
                row["PG_Hit"] = compute_pointing_game(sal, gt)
                row["EBPG"] = compute_EBPG(sal, gt)

            sal_path = os.path.join(SAL_MAPS_DIR, f"frame{frame_idx}_track{int(tid)}.png")
            plt.imsave(sal_path, sal, cmap="jet")
            row["saliency_map_path"] = sal_path

        results.append(row)

# ======================================================
# GUARDAR CSV
# ======================================================
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "gcame_human_sort_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"CSV guardado en {csv_path}")
