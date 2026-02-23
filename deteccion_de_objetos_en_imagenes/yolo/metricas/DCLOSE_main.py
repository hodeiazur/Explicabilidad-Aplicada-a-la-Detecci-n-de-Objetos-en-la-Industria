import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from scipy.stats import pearsonr

# ======================================================
#  CONFIG
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device)

TARGET_SIZE = 640  # YOLO requirement


# ======================================================
# IOU
# ======================================================
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)


# ======================================================
#  PREDICT para YOLO (acepta solo 640x640)
# ======================================================
def predict(model, img_rgb_resized):
    """
    YOLO SOLO trabaja bien con tamaño 640x640.
    Por eso la imagen YA VIENE REDIMENSIONADA,
    y aquí solo hacemos el forward.
    """
    img = img_rgb_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(img_tensor, verbose=False)[0]

    return {
        "boxes": result.boxes.xyxy.cpu().numpy().astype(int),
        "scores": result.boxes.conf.cpu().numpy(),
        "labels": result.boxes.cls.cpu().numpy().astype(int)
    }


# ======================================================
#  D-CLOSE GLOBAL
# ======================================================
def dclose_map_global(img_rgb, model, orig_box, orig_class_id, orig_score,
                      n_levels=[100, 200], mask_value=0, iou_thresh=0.3):

    H, W, _ = img_rgb.shape
    saliency_map = np.zeros((H, W), dtype=np.float32)
    img_float = img_rgb.astype(np.float32) / 255.0

    for n_segments in n_levels:
        segments = slic(img_float, n_segments=n_segments, compactness=10, start_label=0)

        for seg_val in np.unique(segments):
            seg_mask = (segments == seg_val)
            perturbed = img_rgb.copy()
            perturbed[seg_mask] = mask_value

            out_pert = predict(model, perturbed)
            boxes_p, scores_p, labels_p = out_pert["boxes"], out_pert["scores"], out_pert["labels"]

            best_iou, best_score = 0.0, 0.0
            for b, s, l in zip(boxes_p, scores_p, labels_p):
                if l != orig_class_id: 
                    continue
                this_iou = iou(orig_box, b)
                if this_iou > best_iou:
                    best_iou, best_score = this_iou, s

            pred_score_pert = best_score if best_iou >= iou_thresh else 0.0
            delta = orig_score - pred_score_pert
            saliency_map[seg_mask] += delta

    saliency_map -= saliency_map.min()
    saliency_map /= (saliency_map.max() + 1e-8)
    return saliency_map


# ======================================================
#  MÉTRICAS
# ======================================================
def deletion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, mask_value=0, n_levels=[100,200]):
    H, W = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=n_levels[0], compactness=10, start_label=0)

    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = img_rgb.copy()
    c_scores = []
    s_scores = []

    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])

        perturbed[seg_mask] = mask_value
        out_pert = predict(model, perturbed)

        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(
            out_pert["boxes"], out_pert["scores"], out_pert["labels"]
        ):
            if l != orig_class_id: continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2:
        return 0.0

    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])
    s = np.array(s_scores[:-1])

    if np.std(v)==0 or np.std(s)==0:
        return 0.0
    corr,_ = pearsonr(v,s)
    return corr


def insertion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, blur_kernel=21, n_levels=[100,200]):
    H, W = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=n_levels[0], compactness=10, start_label=0)

    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), 0)
    c_scores = []
    s_scores = []

    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = img_rgb[seg_mask]

        out_pert = predict(model, perturbed)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(
            out_pert["boxes"], out_pert["scores"], out_pert["labels"]
        ):
            if l != orig_class_id: continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2:
        return 0.0

    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    s = np.array(s_scores[1:])

    if np.std(v)==0 or np.std(s)==0:
        return 0.0
    corr,_ = pearsonr(v,s)
    return corr


def sparsity(sal_map):
    return 1.0 / (sal_map.mean() + 1e-8)


def compute_pointing_game(sal_map, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    max_y, max_x = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    hit = (x1 <= max_x <= x2) and (y1 <= max_y <= y2)
    return 1 if hit else 0


def compute_EBPG(sal_map, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    S = sal_map.astype(np.float32)
    S_norm = S / (S.sum() + 1e-12)
    sub = S_norm[y1:y2+1, x1:x2+1]
    return float(sub.sum())


def match_prediction_to_gt(pred_box, gt_boxes, iou_thresh=0.5):
    best_gt, best_iou_val = None, 0.0
    for gt in gt_boxes:
        val = iou(pred_box, gt)
        if val > best_iou_val:
            best_iou_val = val
            best_gt = gt
    return best_gt if best_iou_val >= iou_thresh else None


# ======================================================
#  GT loader COCO
# ======================================================
def load_coco_gt(json_path, image_filename):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    img_id = None
    for img in coco["images"]:
        if img["file_name"] == image_filename:
            img_id = img["id"]
            break

    if img_id is None:
        raise ValueError(f"La imagen '{image_filename}' no está en el JSON COCO.")

    gt_boxes, gt_labels = [], []
    for ann in coco["annotations"]:
        if ann["image_id"] == img_id:
            x,y,w,h = ann["bbox"]
            gt_boxes.append([int(x),int(y),int(x+w),int(y+h)])
            gt_labels.append(int(ann["category_id"]))

    return gt_boxes, gt_labels


# ======================================================
#  MAIN
# ======================================================
img_path = "imagenes_testear/rgb_0000.png"
json_path = "instances_default_rgb_0000.json"

gt_boxes, gt_labels = load_coco_gt(json_path, img_path)

output_dir = "DCLOSE-METRICAS-RGB_0000"
os.makedirs(output_dir, exist_ok=True)

# --- Cargar imagen ---
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Resize obligatorio para YOLO ---
H0, W0 = img_rgb.shape[:2]
img_resized = cv2.resize(img_rgb, (TARGET_SIZE, TARGET_SIZE))

# --- Ajustar GT a nuevo tamaño ---
scale_x = TARGET_SIZE / W0
scale_y = TARGET_SIZE / H0
gt_boxes_rescaled = [
    [int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)]
    for (x1,y1,x2,y2) in gt_boxes
]

# --- Inferencia ---
out = predict(model, img_resized)
boxes = out["boxes"]
scores = out["scores"]
classes = out["labels"]

# ======================================================
#  Loop D-CLOSE
# ======================================================
colors = {
    0:(255,0,0),1:(0,255,0),2:(0,0,255),3:(255,255,0),
    4:(255,0,255),5:(0,255,255)
}

results = []

for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
    if score < 0.5:
        continue

    sal_map = dclose_map_global(img_resized, model, box, cls, score)

    gt_box = match_prediction_to_gt(box, gt_boxes_rescaled)

    if gt_box is not None:
        pg = compute_pointing_game(sal_map, gt_box)
        ebpg = compute_EBPG(sal_map, gt_box)
    else:
        pg = 0
        ebpg = 0.0

    dc = deletion_correlation(img_resized, model, box, cls, score, sal_map)
    ic = insertion_correlation(img_resized, model, box, cls, score, sal_map)
    sp = sparsity(sal_map)

    results.append({
        "Object": i+1,
        "Class": cls,
        "Score": score,
        "DC": dc,
        "IC": ic,
        "Sparsity": sp,
        "PG_Hit": pg,
        "EBPG": ebpg
    })

    # --- Visualización ---
    x1,y1,x2,y2 = box
    img_show = img_resized.copy()
    color = colors.get(cls,(255,255,255))
    cv2.rectangle(img_show,(x1,y1),(x2,y2),color,2)
    cv2.putText(img_show,f"cls {cls}: {score:.2f}",(x1,max(10,y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    plt.figure(figsize=(6,6))
    plt.imshow(img_show)
    plt.imshow(sal_map,cmap='jet',alpha=0.5)
    plt.axis('off')
    save_path = os.path.join(output_dir,f"yolo_dclose_obj{i+1}_class{cls}.png")
    plt.savefig(save_path,bbox_inches='tight',pad_inches=0)
    plt.close()

# --- Guardar CSV ---
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "dclose_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"CSV guardado en {csv_path}")

