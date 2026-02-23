import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from gcame import GCAME
import json
import pandas as pd
from skimage.segmentation import slic
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================
# Función IOU
# ===========================================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-8)

# ===========================================
# Wrapper predict() para Faster R-CNN
# ===========================================
def predict(model, img_numpy):
    t = transforms.ToTensor()(img_numpy).to(DEVICE)
    with torch.no_grad():
        out = model([t])[0]
    return out

# ===========================================
# Normalizar saliency
# ===========================================
def normalize_saliency(hm):
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return hm


# ===========================================
# MÉTRICAS
# ===========================================
# ----------------------------
# Helpers adicionales
# ----------------------------
def crop_with_padding(img, box, pad=0.2):
    H, W = img.shape[:2]
    x1,y1,x2,y2 = box
    w = x2-x1; h = y2-y1
    px = int(w * pad); py = int(h * pad)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(W-1, x2 + px); ny2 = min(H-1, y2 + py)
    return (nx1, ny1, nx2, ny2)

def adaptive_n_segments(box, min_seg=60, max_seg=500):
    x1,y1,x2,y2 = box
    area = max(1, (x2-x1)*(y2-y1))
    # heurística: 1 segment por ~400 px2
    seg = int(np.clip(area / 400, min_seg, max_seg))
    return seg

def local_mean_background(img_rgb, box, pad_box=None):
    # calcula media de píxeles fuera del bbox (o en pad box border) para usar como mask_value
    H,W = img_rgb.shape[:2]
    if pad_box is None:
        pad_box = (0,0,W-1,H-1)
    mask = np.ones((H,W), dtype=bool)
    x1,y1,x2,y2 = box
    mask[y1:y2+1, x1:x2+1] = False
    if mask.sum() == 0:
        return int(img_rgb.mean(axis=(0,1)))
    bg = img_rgb[mask].reshape(-1,3).mean(axis=0)
    return tuple([int(float(v)) for v in bg])  # BGR-like tuple

# ----------------------------
# Normalización local del mapa (dentro del bbox)
# ----------------------------
def normalize_saliency_in_box(sal_map, box, eps=1e-8):
    x1,y1,x2,y2 = map(int, box)
    sub = sal_map[y1:y2+1, x1:x2+1]
    sub = sub - sub.min()
    denom = sub.max() + eps
    if denom <= 0:
        sub_norm = np.zeros_like(sub)
    else:
        sub_norm = sub / denom
    # reconstruir mapa completo donde fuera del bbox es 0
    S = np.zeros_like(sal_map, dtype=np.float32)
    S[y1:y2+1, x1:x2+1] = sub_norm
    return S

# ----------------------------
# Deletion correlation (bbox-local)
# ----------------------------
def deletion_correlation_bbox(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map,
                              pad=0.2, n_segments=None, blur_on_mask=False):
    H,W = img_rgb.shape[:2]
    # crop bbox with padding -- we will segment in this crop
    bx = crop_with_padding(img_rgb, orig_box, pad=pad)
    cx1,cy1,cx2,cy2 = bx
    crop = img_rgb[cy1:cy2+1, cx1:cx2+1].astype(np.float32) / 255.0
    if n_segments is None:
        n_segments = adaptive_n_segments((cx1,cy1,cx2,cy2))
    segments = slic(crop, n_segments=n_segments, compactness=10, start_label=0)

    # normalize saliency inside bbox (and place in full-size map)
    sal_local = normalize_saliency_in_box(sal_map, orig_box)

    # compute per-seg scores using sal_local but restricted to crop coordinates
    seg_vals = np.unique(segments)
    seg_scores = []
    for v in seg_vals:
        mask = (segments == v)
        # map mask coords to full image
        full_mask = np.zeros((H,W), dtype=bool)
        full_mask[cy1:cy2+1, cx1:cx2+1][mask] = True
        seg_scores.append(sal_local[full_mask].mean() if full_mask.sum()>0 else 0.0)
    sorted_idx = np.argsort(seg_scores)[::-1]

    # prepare masked image baseline and mask value: use local background mean (prevents black OOD)
    mask_value = local_mean_background(img_rgb, orig_box, pad_box=bx)  # RGB tuple
    # create a float copy
    perturbed = img_rgb.copy()

    c_scores = []
    s_scores = []

    # iterate perturbations (most salient -> less)
    for idx in sorted_idx:
        # build full mask for this segment
        mask = (segments == seg_vals[idx])
        full_mask = np.zeros((H,W), dtype=bool)
        full_mask[cy1:cy2+1, cx1:cx2+1][mask] = True

        s_scores.append(seg_scores[idx])
        # apply mask_value or blur region (option)
        if blur_on_mask:
            # gaussian blur the bbox region then copy only masked pixels
            blurred = cv2.GaussianBlur(perturbed, (21,21), 0)
            perturbed[full_mask] = blurred[full_mask]
        else:
            perturbed[full_mask] = mask_value

        out = predict(model, perturbed)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
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
    corr, _ = pearsonr(v, s)
    return corr

# ----------------------------
# Insertion correlation (bbox-local)
# ----------------------------
def insertion_correlation_bbox(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map,
                               pad=0.2, n_segments=None, blur_kernel=21):
    H,W = img_rgb.shape[:2]
    bx = crop_with_padding(img_rgb, orig_box, pad=pad)
    cx1,cy1,cx2,cy2 = bx
    crop = img_rgb[cy1:cy2+1, cx1:cx2+1].astype(np.float32) / 255.0
    if n_segments is None:
        n_segments = adaptive_n_segments((cx1,cy1,cx2,cy2))
    segments = slic(crop, n_segments=n_segments, compactness=10, start_label=0)

    sal_local = normalize_saliency_in_box(sal_map, orig_box)
    seg_vals = np.unique(segments)
    seg_scores = []
    for v in seg_vals:
        mask = (segments == v)
        full_mask = np.zeros((H,W), dtype=bool)
        full_mask[cy1:cy2+1, cx1:cx2+1][mask] = True
        seg_scores.append(sal_local[full_mask].mean() if full_mask.sum()>0 else 0.0)
    sorted_idx = np.argsort(seg_scores)[::-1]

    # start from globally blurred image
    perturbed = cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), 0)
    c_scores = []
    s_scores = []

    for idx in sorted_idx:
        mask = (segments == seg_vals[idx])
        full_mask = np.zeros((H,W), dtype=bool)
        full_mask[cy1:cy2+1, cx1:cx2+1][mask] = True

        s_scores.append(seg_scores[idx])
        # reveal original pixels for this segment
        perturbed[full_mask] = img_rgb[full_mask]

        out = predict(model, perturbed)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
            if l != orig_class_id: continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2: return 0.0

    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    s = np.array(s_scores[1:])
    if np.std(v)==0 or np.std(s)==0: return 0.0
    corr, _ = pearsonr(v, s)
    return corr

# ----------------------------
# Sparsity corregida (usar mapa normalizado dentro bbox)
# ----------------------------
def sparsity_box(sal_map, box):
    sal_in = sal_map[box[1]:box[3]+1, box[0]:box[2]+1]
    sal_in = sal_in - sal_in.min()
    denom = sal_in.mean() + 1e-8
    return 1.0 / denom

def compute_pointing_game(sal_map, gt_box):
    """
    sal_map: (H,W) numpy array normalizado [0..1]
    gt_box: [x1,y1,x2,y2] coordenadas de GT
    """
    x1, y1, x2, y2 = map(int, gt_box)

    # encontrar el punto máximo de la saliency
    max_pos = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    max_y, max_x = max_pos   # (fila, columna)

    # chequeo si está dentro de la GT
    hit = (x1 <= max_x <= x2) and (y1 <= max_y <= y2)
    return 1 if hit else 0


def compute_EBPG(sal_map, gt_box):
    """
    EBPG = suma energía normalizada dentro de GT.
    """
    x1, y1, x2, y2 = map(int, gt_box)

    # normalización S' = S / sum(S)
    S = sal_map.astype(np.float32)
    S_norm = S / (S.sum() + 1e-12)

    # recorte a GT
    sub = S_norm[y1:y2+1, x1:x2+1]
    return float(sub.sum())

# ============================================================
# MATCH con GT
# ============================================================
def match_prediction_to_gt(pred_box, gt_boxes, thr=0.5):
    best_i = -1
    best_val = 0
    for i, gt in enumerate(gt_boxes):
        v = iou(pred_box, gt)
        if v > best_val:
            best_val = v
            best_i = i
    return gt_boxes[best_i] if best_val >= thr else None


# ============================================================
# FUNCIÓN PARA CARGAR GT COCO
# ============================================================
def load_coco_gt(json_path, image_filename):
    with open(json_path, "r") as f:
        coco = json.load(f)

    img_id = next(img["id"] for img in coco["images"] if img["file_name"] == image_filename)

    gt_boxes, gt_labels = [], []
    for ann in coco["annotations"]:
        if ann["image_id"] == img_id:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([int(x), int(y), int(x+w), int(y+h)])
            gt_labels.append(int(ann["category_id"]))

    return gt_boxes, gt_labels

# ============================
# CONFIG
# ============================
MODEL_PATH = "fasterrcnn_trained_4.pth"
IMG_PATH = "imagenes_testear/00901.jpg"
JSON_PATH = "imagenes_testear/instances_default_00901.json"
OUTDIR = "METRICAS_GCAME_00901_LAYERS_ROI2"
os.makedirs(OUTDIR, exist_ok=True)

# ============================
# CARGAR MODELO
# ============================
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=12)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ============================
# CARGAR IMAGEN
# ============================
img = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

out = predict(model, img_rgb)
print("Pred first scores:", out["scores"].cpu().numpy()[:5])

boxes = out["boxes"].cpu().numpy().astype(int)
scores = out["scores"].cpu().numpy()
labels = out["labels"].cpu().numpy()

# ============================
# CARGAR GT
# ============================
gt_boxes, gt_labels = load_coco_gt(JSON_PATH, IMG_PATH)

# ============================
# CONFIG GCAME
# ============================
target_layers = [
    "backbone.body.layer4"  # o layer3.5 para mapas más finos
]



gcame = GCAME(model, target_layers=target_layers, arch="fasterrcnn")

results = []

# ====================================================
# LOOP POR OBJETO DETECTADO
# ====================================================
for i, (box, cls, score) in enumerate(zip(boxes, labels, scores)):

    if score < 0.5:
        continue

    x1,y1,x2,y2 = box
    print(f"Obj {i}: cls={cls} score={score:.3f}")

    # --- GCAME MAP ---
    img_tensor = transforms.ToTensor()(img_rgb).to(DEVICE)  # [C,H,W] sin batch
    heat = gcame(img_tensor, box, obj_idx=i)
    sal = cv2.resize(heat, (img_rgb.shape[1], img_rgb.shape[0]))
    sal = normalize_saliency(sal)

    # debug saves
    cx = crop_with_padding(img_rgb, box, pad=0.2)
    cx1,cy1,cx2,cy2 = cx
    crop_img = img_rgb[cy1:cy2+1, cx1:cx2+1]
    crop_sal = sal[cy1:cy2+1, cx1:cx2+1]
    cv2.imwrite(f"{OUTDIR}/obj{i}_crop.jpg", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{OUTDIR}/obj{i}_crop_sal.jpg", cv2.applyColorMap(np.uint8(255*normalize_saliency(crop_sal)), cv2.COLORMAP_JET))

    # =============================
    # GUARDAR IMÁGENES DE SALIENCY
    # =============================
    # 1) Heatmap original (color)
    heatmap_uint8 = np.uint8(255 * sal)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 2) Overlay con la imagen original
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    # Dibujar bounding box
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(overlay, f"obj {i} | cls {cls} | {score:.2f}", 
                (x1, max(10,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Guardar imágenes
    cv2.imwrite(f"{OUTDIR}/obj{i}_cls{cls}_heatmap.jpg", cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{OUTDIR}/obj{i}_cls{cls}_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))



    # =============================
    # MÉTRICAS
    # =============================
    dc = deletion_correlation_bbox(
        img_rgb, model, box, cls, score, sal,
        pad=0.20,          # padding del bbox
        n_segments=None,   # adaptativo
        blur_on_mask=False # o True si quieres más suave
    )

    ic = insertion_correlation_bbox(
            img_rgb, model, box, cls, score, sal,
            pad=0.20,
            n_segments=None,
            blur_kernel=21
        )

    sp = sparsity_box(sal, box)

    gt_box = match_prediction_to_gt(box, gt_boxes)

    if gt_box is not None:
        pg = compute_pointing_game(sal, gt_box)
        ebpg = compute_EBPG(sal, gt_box)
    else:
        pg = 0
        ebpg = 0

    results.append({
        "Object": i,
        "Class": cls,
        "Score": score,
        "DC": dc,
        "IC": ic,
        "Sparsity": sp,
        "PG_Hit": pg,
        "EBPG": ebpg
    })

# =============================
# GUARDAR CSV
# =============================
df = pd.DataFrame(results)
df.to_csv(f"{OUTDIR}/metrics.csv", index=False)
print("CSV guardado.")

