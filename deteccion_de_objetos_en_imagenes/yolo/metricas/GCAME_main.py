
import os
import cv2
import json
import time
import math
import numpy as np
import pandas as pd
import csv
from ultralytics import YOLO
from gcame2 import GCAME 

# ========= Métricas: utilidades =========
from scipy.stats import pearsonr
from skimage.segmentation import slic

def iou(boxA, boxB):
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, float(boxA[2]) - float(boxA[0])) * max(0.0, float(boxA[3]) - float(boxA[1]))
    areaB = max(0.0, float(boxB[2]) - float(boxB[0])) * max(0.0, float(boxB[3]) - float(boxB[1]))
    denom = areaA + areaB - inter + 1e-12
    return inter / denom if denom > 0 else 0.0

def predict_ultralytics(model, img_bgr, conf=None):
    """Ejecuta YOLO sobre un numpy BGR y devuelve dict con boxes, scores, labels (numpy)."""
    if conf is None:
        results = model(img_bgr, verbose=False)
    else:
        results = model(img_bgr, conf=conf, iou=0.7, verbose=False)
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return {"boxes": np.zeros((0,4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int32)}
    boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = r0.boxes.conf.cpu().numpy().astype(np.float32)
    labels = r0.boxes.cls.cpu().numpy().astype(np.int32)
    return {"boxes": boxes, "scores": scores, "labels": labels}

def deletion_correlation(img_rgb, model, orig_box, orig_class_id, sal_map,
                         mask_value=0, n_segments=120, conf=None, iou_match_thresh=0.3):
    """
    Elimina segmentos SLIC desde los más importantes (según sal_map) y
    mide la caída de score de la predicción que mejor coincide con (orig_box, orig_class_id).
    Devuelve Pearson corr entre: v (delta de score) y s (importancia del segmento eliminado).
    """
    H, W = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float32) / 255.0
    segments = slic(img_float, n_segments=n_segments, compactness=10, start_label=0)
    seg_vals = np.unique(segments)

    # importancia por segmento
    seg_scores = [sal_map[segments == v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]  # más importante primero

    perturbed = img_rgb.copy()
    c_scores, s_scores = [], []

    # score inicial
    out0 = predict_ultralytics(model, perturbed, conf=conf)
    best_iou, best_score = 0.0, 0.0
    for b, s, l in zip(out0["boxes"], out0["scores"], out0["labels"]):
        if l != orig_class_id: 
            continue
        ii = iou(orig_box, b)
        if ii > best_iou:
            best_iou, best_score = ii, float(s)
    c_scores.append(best_score if best_iou > iou_match_thresh else 0.0)

    # eliminar segmentos
    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = mask_value  # "apagar" región

        out_pert = predict_ultralytics(model, perturbed, conf=conf)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out_pert["boxes"], out_pert["scores"], out_pert["labels"]):
            if l != orig_class_id: 
                continue
            ii = iou(orig_box, b)
            if ii > best_iou:
                best_iou, best_score = ii, float(s)
        c_scores.append(best_score if best_iou > iou_match_thresh else 0.0)

    if len(c_scores) < 3:
        return 0.0
    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])   # caída de score por paso
    s = np.array(s_scores)                                 # importancia del segmento eliminado
    if np.std(v) == 0 or np.std(s) == 0:
        return 0.0
    corr, _ = pearsonr(v, s)
    return float(corr)

def insertion_correlation(img_rgb, model, orig_box, orig_class_id, sal_map,
                          blur_kernel=21, n_segments=120, conf=None, iou_match_thresh=0.3):
    """
    Parte de una imagen borrosa y "reconstruye" por segmentos desde los más importantes;
    mide la subida de score y devuelve corr(v,s) análoga a deletion.
    """
    H, W = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float32) / 255.0
    segments = slic(img_float, n_segments=n_segments, compactness=10, start_label=0)
    seg_vals = np.unique(segments)

    seg_scores = [sal_map[segments == v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), 0)
    c_scores, s_scores = [], []

    # score inicial (borroso)
    out0 = predict_ultralytics(model, perturbed, conf=conf)
    best_iou, best_score = 0.0, 0.0
    for b, s, l in zip(out0["boxes"], out0["scores"], out0["labels"]):
        if l != orig_class_id: 
            continue
        ii = iou(orig_box, b)
        if ii > best_iou:
            best_iou, best_score = ii, float(s)
    c_scores.append(best_score if best_iou > iou_match_thresh else 0.0)

    # insertar segmentos
    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = img_rgb[seg_mask]

        out_pert = predict_ultralytics(model, perturbed, conf=conf)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out_pert["boxes"], out_pert["scores"], out_pert["labels"]):
            if l != orig_class_id:
                continue
            ii = iou(orig_box, b)
            if ii > best_iou:
                best_iou, best_score = ii, float(s)
        c_scores.append(best_score if best_iou > iou_match_thresh else 0.0)

    if len(c_scores) < 3:
        return 0.0
    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])   # subida por paso
    s = np.array(s_scores)                                 # importancia insertada
    if np.std(v) == 0 or np.std(s) == 0:
        return 0.0
    corr, _ = pearsonr(v, s)
    return float(corr)

def sparsity(sal_map):
    """Para sal_map normalizado a [0,1], sparsity = 1/mean(S)."""
    return float(1.0 / (float(sal_map.mean()) + 1e-8))

def compute_pointing_game(sal_map, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    max_pos = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    max_y, max_x = max_pos
    hit = (x1 <= max_x <= x2) and (y1 <= max_y <= y2)
    return int(hit)

def compute_EBPG(sal_map, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    S = sal_map.astype(np.float32)
    S_norm = S / (S.sum() + 1e-12)
    sub = S_norm[max(0,y1):max(0,y2)+1, max(0,x1):max(0,x2)+1]
    return float(sub.sum())

def load_coco_gt(json_path, image_filename):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    img_id = None
    for img in coco["images"]:
        if img["file_name"] == image_filename:
            img_id = img["id"]; break
    if img_id is None:
        raise ValueError(f"No se ha encontrado '{image_filename}' en el JSON COCO.")
    gt_boxes, gt_labels = [], []
    for ann in coco["annotations"]:
        if ann["image_id"] == img_id:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([int(x), int(y), int(x+w), int(y+h)])
            gt_labels.append(int(ann["category_id"]))
    return gt_boxes, gt_labels

def match_prediction_to_gt(pred_box, gt_boxes, iou_thresh=0.5):
    best_gt, best_iou_val = None, 0.0
    for gt in gt_boxes:
        val = iou(pred_box, gt)
        if val > best_iou_val:
            best_iou_val, best_gt = val, gt
    if best_gt is None or best_iou_val < iou_thresh:
        return None
    return best_gt

def write_metrics_csv(save_csv, rows):
    fieldnames = [
        "image", "idx", "class_id", "class_name", "score",
        "x1", "y1", "x2", "y2",
        "sparsity", "deletion_corr", "insertion_corr",
        "PG_hit", "EBPG", "gt_iou_match",
        "t_saliency_s", "t_perturb_s", "overlay_path"
    ]
    file_exists = os.path.exists(save_csv)
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    with open(save_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

# ========= Tu flujo original + métricas =========

# --------- Config ---------
img_path = "imagenes_testear/rgb_0000.png"
model_path = "best.pt"
conf_threshold = 0.5
save_dir = "mapas_calor_gcame_oficiales_con_bbox_metricas_BUENO"
colormap = cv2.COLORMAP_TURBO
alpha_overlay = 0.6

# (Opcional) JSON COCO para PG/EBPG; si no tienes, pon None
coco_json_path = "instances_default_rgb_0000.json"

# Parámetros de métricas (puedes ajustarlos)
N_SEGMENTS = 120
BLUR_KERNEL = 21
IOU_MATCH_THRESH = 0.50
CSV_OUT = os.path.join(save_dir, "gcame_metrics.csv")
IMG_SIZE = 640
os.makedirs(save_dir, exist_ok=True)

# --------- Cargar modelo y predecir ---------
model = YOLO(model_path)
results = model(img_path)  # predicción rápida (no afecta a GCAME)
res = results[0]
img_bgr = cv2.imread(img_path)
H, W = img_bgr.shape[:2]

# Extraer info de predicciones
boxes = res.boxes
pred_bboxes = boxes.xyxy.cpu().numpy()   # (N,4) [x1,y1,x2,y2]
pred_scores = boxes.conf.cpu().numpy()   # (N,)
pred_labels = boxes.cls.cpu().numpy()    # (N,)
names = res.names if hasattr(res, "names") else model.names  # dict id->name
base_name = os.path.splitext(os.path.basename(img_path))[0]

# Filtrado por confianza
keep = pred_scores >= conf_threshold
pred_bboxes = pred_bboxes[keep]
pred_scores = pred_scores[keep]
pred_labels = pred_labels[keep]

print(f"Objetos a explicar: {len(pred_bboxes)}")

# --------- Instanciar GCAME ---------
gcame = GCAME(model, arch="ultralytics", img_size=(IMG_SIZE, IMG_SIZE))
img_bgr_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

# (Opcional) GT para PG/EBPG
gt_boxes, gt_labels = ([], [])
if coco_json_path is not None:
    gt_boxes, gt_labels = load_coco_gt(coco_json_path, os.path.basename(img_path))

# Acumular métricas por objeto para CSV
rows = []

# --------- Generar y guardar overlay por objeto (con bbox) ---------
def class_color(cls_id: int):
    np.random.seed(cls_id + 12345)
    c = np.random.randint(64, 256, size=3, dtype=np.uint8)
    return (int(c[0]), int(c[1]), int(c[2]))  # BGR

for i, (box, score, cid) in enumerate(zip(pred_bboxes, pred_scores, pred_labels)):
    cls_id = int(cid)
    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
    print(f"[{i:02d}] {cls_name}  conf={score:.3f}  box={box.tolist()}")
    sx = IMG_SIZE / W
    sy = IMG_SIZE / H

    box_rescaled = np.array([
        box[0] * sx,
        box[1] * sy,
        box[2] * sx,
        box[3] * sy
    ])

    # --- Saliency con G‑CAME ---
    t0 = time.time()

    sal_640 = gcame.forward_ultralytics_yolo(
        img_bgr_resized,
        box_rescaled,
        cls_id=cls_id
    )

    sal = cv2.resize(sal_640, (W, H), interpolation=cv2.INTER_LINEAR)

    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)

    t_sal = time.time() - t0


    # --- Overlay saliency ---
    heat = (sal * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, colormap)
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha_overlay, heat_color, alpha_overlay, 0)

    # --- Dibujar bbox + etiqueta sobre el overlay ---
    x1, y1, x2, y2 = map(int, box)
    color = class_color(cls_id)
    thickness = max(2, int(round(0.003 * (H + W))))
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

    label = f"{cls_name} {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, 0.0009 * (H + W))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    tx1, ty1 = x1, max(0, y1 - th - baseline - 2)
    tx2, ty2 = x1 + tw + 6, y1
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(overlay, label, (tx1 + 3, y1 - baseline - 2), font, font_scale,
                (0, 0, 0), max(1, thickness - 1), cv2.LINE_AA)

    # --- Guardar overlay ---
    out_name = f"{base_name}_idx{i:02d}_{cls_name}_conf{score:.3f}.png"
    out_path = os.path.join(save_dir, out_name)
    cv2.imwrite(out_path, overlay)

    # --- Métricas ---
    m_sparsity = sparsity(sal)

    t1 = time.time()
    del_corr = deletion_correlation(
        img_rgb=img_bgr, model=model, orig_box=box, orig_class_id=cls_id,
        sal_map=sal, mask_value=0, n_segments=N_SEGMENTS,
        conf=conf_threshold, iou_match_thresh=IOU_MATCH_THRESH
    )
    ins_corr = insertion_correlation(
        img_rgb=img_bgr, model=model, orig_box=box, orig_class_id=cls_id,
        sal_map=sal, blur_kernel=BLUR_KERNEL, n_segments=N_SEGMENTS,
        conf=conf_threshold, iou_match_thresh=IOU_MATCH_THRESH
    )
    t_pert = time.time() - t1

    # PG / EBPG con GT si existe emparejamiento IoU>=0.5
    pg_hit, ebpg_val, gt_iou_match = np.nan, np.nan, np.nan
    if len(gt_boxes) > 0:
        gt_match = match_prediction_to_gt(box, gt_boxes, iou_thresh=0.5)
        if gt_match is not None:
            pg_hit = compute_pointing_game(sal, gt_match)
            ebpg_val = compute_EBPG(sal, gt_match)
            gt_iou_match = iou(box, gt_match)

    rows.append({
        "image": base_name,
        "idx": i,
        "class_id": cls_id,
        "class_name": cls_name,
        "score": float(score),
        "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
        "sparsity": m_sparsity,
        "deletion_corr": del_corr,
        "insertion_corr": ins_corr,
        "PG_hit": pg_hit,
        "EBPG": ebpg_val,
        "gt_iou_match": gt_iou_match,
        "t_saliency_s": t_sal,
        "t_perturb_s": t_pert,
        "overlay_path": out_path
    })

write_metrics_csv(CSV_OUT, rows)
print(f"Métricas guardadas en: {os.path.abspath(CSV_OUT)}")

print(f"Overlays guardados en: {os.path.abspath(save_dir)}")

# --- Guardar CSV con métricas (append si existe) ---
df = pd.DataFrame(rows)
csv_mode = "a" if os.path.exists(CSV_OUT) else "w"
csv_header = not os.path.exists(CSV_OUT)
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
df.to_csv(CSV_OUT, mode=csv_mode, header=csv_header, index=False, float_format="%.6f")
print(f"Métricas guardadas en: {os.path.abspath(CSV_OUT)}")
print(df[["class_name","score","sparsity","deletion_corr","insertion_corr","PG_hit","EBPG"]])

