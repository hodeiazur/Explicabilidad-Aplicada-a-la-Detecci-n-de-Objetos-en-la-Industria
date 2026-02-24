import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from gcame import GCAME
from skimage.segmentation import slic
from scipy.stats import pearsonr

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best.pt"

IMAGE_DIR  = "cam_ta1_ws2/"
OUTPUT_DIR = "GCAME_SORT_RESULTS_dobleSigma_metricas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_SIZE = (640, 640)

CONF_DET  = 0.5

GT_JSON = "ground_truth_sinMujerAtras.json"

YOLO_HUMAN_CLASS_ID = 0
COCO_HUMAN_CATEGORY_ID = 1

CSV_OUT = os.path.join(OUTPUT_DIR, "gcame_metrics.csv")

# =========================
# Cargar modelo
# =========================
model = YOLO(MODEL_PATH)
model.to(DEVICE)

gcame = GCAME(model, arch="ultralytics", img_size=FIXED_SIZE)

# =========================
# Cargar GT una sola vez
# =========================
with open(GT_JSON, "r") as f:
    coco = json.load(f)

def get_gt_for_image(image_name):
    img_id = None
    for img in coco["images"]:
        if img["file_name"] == image_name:
            img_id = img["id"]
            break

    if img_id is None:
        return []

    boxes = []
    for ann in coco["annotations"]:
        if ann["image_id"] == img_id and ann["category_id"] == COCO_HUMAN_CATEGORY_ID:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x+w, y+h])
    return boxes


def iou(boxA, boxB):
    """IoU entre dos boxes [x1,y1,x2,y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-12)

def predict(img_np):
    """Wrapper simple de tu YOLO para métricas"""
    results = model.predict(img_np, conf=CONF_DET, device=DEVICE, verbose=False)
    boxes, scores, labels = [], [], []
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return {"boxes": torch.tensor([]), "scores": torch.tensor([]), "labels": torch.tensor([])}
    for b,s,l in zip(r.boxes.xyxy.cpu().numpy(),
                     r.boxes.conf.cpu().numpy(),
                     r.boxes.cls.cpu().numpy().astype(int)):
        boxes.append(b)
        scores.append(s)
        labels.append(l)
    return {"boxes": torch.tensor(boxes), "scores": torch.tensor(scores), "labels": torch.tensor(labels)}

def deletion_correlation(img_rgb, orig_box, orig_class_id, sal_map, n_segments=200):
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

def insertion_correlation(img_rgb, orig_box, orig_class_id, sal_map, n_segments=200):
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

def compute_pointing_game(sal_map, gt_box):
    x1,y1,x2,y2 = map(int, gt_box)
    y,x = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    return int(x1<=x<=x2 and y1<=y<=y2)

def compute_EBPG(sal_map, gt_box):
    x1,y1,x2,y2 = map(int, gt_box)
    S = sal_map / (sal_map.sum() + 1e-12)
    return float(S[y1:y2+1, x1:x2+1].sum())


# =========================
# Procesar carpeta
# =========================
rows = []

image_paths = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png",".jpg",".jpeg"))
])

print(f"Imágenes encontradas: {len(image_paths)}")

for img_path in image_paths:

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\nProcesando {base_name}")

    img_bgr = cv2.imread(img_path)
    H, W = img_bgr.shape[:2]

    # =========================
    # DETECCIÓN (sin letterbox)
    # =========================
    results = model.predict(
        img_bgr,
        conf=CONF_DET,
        device=DEVICE,
        verbose=False
    )

    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        print("  Sin detecciones")
        continue

    boxes  = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy().astype(int)

    # Filtrar solo humanos
    mask = (labels == YOLO_HUMAN_CLASS_ID) & (scores >= CONF_DET)

    boxes  = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        print("  Sin humanos detectados")
        continue

    print(f"  Humanos detectados: {len(boxes)}")

    # =========================
    # Resize directo a 640x640
    # =========================
    img_resized = cv2.resize(img_bgr, FIXED_SIZE)
    sx = FIXED_SIZE[0] / W
    sy = FIXED_SIZE[1] / H

    gt_boxes = get_gt_for_image(os.path.basename(img_path))

    # =========================
    # Para cada humano
    # =========================
    for i, (box, score) in enumerate(zip(boxes, scores)):

        # Reescalar bbox al espacio 640x640
        box_rescaled = np.array([
            box[0] * sx,
            box[1] * sy,
            box[2] * sx,
            box[3] * sy
        ])

        # =========================
        # SALIENCY GCAME
        # =========================
        t0 = time.time()
        sal_640 = gcame.forward_ultralytics_yolo(
            img_resized,
            box_rescaled,
            cls_id=YOLO_HUMAN_CLASS_ID
        )
        sal = cv2.resize(sal_640, (W, H))
        sal = (sal - sal.min()) / (sal.max() + 1e-8)
        t_sal = time.time() - t0

        # =========================
        # Overlay
        # =========================
        heat = (sal * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(img_bgr, 0.4, heat_color, 0.6, 0)
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),2)
        out_name = f"{base_name}_idx{i:02d}_conf{score:.3f}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, overlay)

        # =========================
        # Métricas completas
        # =========================
        sparsity_val = float(1.0 / (sal.mean() + 1e-8))

        # Asociar con GT más cercano
        matched_gt = None
        best_iou = 0.0
        for gt in gt_boxes:
            v = iou(box, gt)
            if v > best_iou:
                best_iou, matched_gt = v, gt
        if matched_gt is None:
            matched_gt = box  # fallback

        deletion_val  = deletion_correlation(img_bgr, matched_gt, YOLO_HUMAN_CLASS_ID, sal)
        insertion_val = insertion_correlation(img_bgr, matched_gt, YOLO_HUMAN_CLASS_ID, sal)
        pg_val        = compute_pointing_game(sal, matched_gt)
        ebpg_val      = compute_EBPG(sal, matched_gt)

        rows.append({
            "image": base_name,
            "idx": i,
            "score": float(score),
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "sparsity": sparsity_val,
            "deletion": deletion_val,
            "insertion": insertion_val,
            "pointing_game": pg_val,
            "EBPG": ebpg_val,
            "t_saliency": t_sal,
            "overlay_path": out_path
        })

# =========================
# Guardar CSV final
# =========================
df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)
print(f"\nCSV guardado en {CSV_OUT}")