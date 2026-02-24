import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from PIL import Image
from ultralytics import YOLO

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import slic
from scipy.stats import pearsonr

# ======================================================
# CONFIGURACIÓN
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "best.pt"

CONF_THRESH = 0.5          # Umbral para YOLO en inferencia
DCL_SCORE_THRESH = 0.6     # Umbral para decidir qué detecciones explicar
HUMAN_CLASS_ID = 0         # ID de la clase "persona" del modelo YOLO
NUM_CLASSES_LIME = 2       # 0 = no-humano, 1 = humano

IMG_DIR = "cam_ta1_ws2/"
GT_JSON = "ground_truth_sinMujerAtras.json"

OUTPUT_DIR = "LIME_YOLO_SORT_RESULTS"
SAL_MAPS_DIR = os.path.join(OUTPUT_DIR, "saliency_maps")
os.makedirs(SAL_MAPS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"DEVICE: {DEVICE}")

# ======================================================
# MODELO YOLO
# ======================================================
model = YOLO(MODEL_PATH)
print("Modelo YOLO cargado")

# ======================================================
# UTILIDADES
# ======================================================
def iou(boxA, boxB):
    """Calcula IoU entre dos cajas [x1,y1,x2,y2]."""
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, (float(boxA[2]) - float(boxA[0]))) * max(0.0, (float(boxA[3]) - float(boxA[1])))
    areaB = max(0.0, (float(boxB[2]) - float(boxB[0]))) * max(0.0, (float(boxB[3]) - float(boxB[1])))
    denom = areaA + areaB - inter + 1e-8
    return inter / denom if denom > 0 else 0.0

def predict_yolo(model, img, conf=0.0):
    """
    Ejecuta YOLO sobre una imagen (numpy HxWxC uint8) y devuelve:
    - boxes: np.array Nx4 (xyxy)
    - scores: np.array N
    - cls_ids: np.array N (int)
    """
    yolo_device = 0 if DEVICE == 'cuda' else 'cpu'
    r = model(img, conf=conf, iou=0.7, max_det=100, device=yolo_device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4)), np.array([]), np.array([], dtype=int)
    return (
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
        r.boxes.cls.cpu().numpy().astype(int)
    )

# ======================================================
# MÉTRICAS
# ======================================================
def deletion_correlation(img, model, orig_box, cls_id, sal):
    """
    Borra progresivamente superpixeles con mayor peso y mide cómo cae la confianza
    del detector en la caja y clase originales. Correlaciona el descenso con la saliencia.
    """
    segments = slic(img.astype(float)/255.0, n_segments=100, start_label=0)
    seg_ids = np.unique(segments)
    scores = [float(sal[segments == s].mean()) for s in seg_ids]
    order = np.argsort(scores)[::-1]

    pert = img.copy()
    confs, salvals = [], []

    for idx in order:
        pert[segments == seg_ids[idx]] = 0
        boxes, scs, cls = predict_yolo(model, pert, conf=0.0)
        best = 0.0
        for b, s, c in zip(boxes, scs, cls):
            if c == cls_id and iou(orig_box, b) > 0.3:
                best = max(best, float(s))
        confs.append(best)
        salvals.append(scores[idx])

    if len(confs) < 2:
        return 0.0
    v = np.diff(confs) * -1.0  # decrementos
    return float(pearsonr(v, salvals[:-1])[0]) if np.std(v) > 0 else 0.0

def insertion_correlation(img, model, orig_box, cls_id, sal):
    """
    Inserta progresivamente superpixeles con mayor peso partiendo de imagen difuminada
    y mide cómo sube la confianza del detector en la caja/clase. Correlaciona el aumento con saliencia.
    """
    segments = slic(img.astype(float)/255.0, n_segments=100, start_label=0)
    seg_ids = np.unique(segments)
    scores = [float(sal[segments == s].mean()) for s in seg_ids]
    order = np.argsort(scores)[::-1]

    pert = cv2.GaussianBlur(img, (21, 21), 0)
    confs, salvals = [], []

    for idx in order:
        pert[segments == seg_ids[idx]] = img[segments == seg_ids[idx]]
        boxes, scs, cls = predict_yolo(model, pert, conf=0.0)
        best = 0.0
        for b, s, c in zip(boxes, scs, cls):
            if c == cls_id and iou(orig_box, b) > 0.3:
                best = max(best, float(s))
        confs.append(best)
        salvals.append(scores[idx])

    if len(confs) < 2:
        return 0.0
    v = np.diff(confs)  # incrementos
    return float(pearsonr(v, salvals[1:])[0]) if np.std(v) > 0 else 0.0

def sparsity(sal):
    return float(1.0 / (sal.mean() + 1e-8))

def pointing_game(sal, gt):
    """
    1 si el máximo de saliencia cae dentro del GT, 0 si no.
    """
    if gt is None:
        return 0
    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    x1, y1, x2, y2 = map(int, gt)
    return int(x1 <= x <= x2 and y1 <= y <= y2)

def ebpg(sal, gt):
    """
    Energy-based pointing game: suma de saliencia normalizada dentro del GT.
    """
    if gt is None:
        return 0.0
    x1, y1, x2, y2 = map(int, gt)
    s = sal / (sal.sum() + 1e-8)
    # recorta por seguridad a límites de la imagen
    H, W = sal.shape
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W - 1, x2), min(H - 1, y2)
    if x2c < x1c or y2c < y1c:
        return 0.0
    return float(s[y1c:y2c+1, x1c:x2c+1].sum())

# ======================================================
# GROUND TRUTH (COCO ÚNICO)
# ======================================================
def load_coco_index(json_path):
    """
    Carga el archivo COCO y devuelve un dict:
      { "filename.jpg": [ {"bbox":[x1,y1,x2,y2], "category_id": id}, ... ] }
    """
    with open(json_path, 'r') as f:
        coco = json.load(f)

    img_by_id = {img["id"]: img for img in coco.get("images", [])}
    by_fname = {}

    for ann in coco.get("annotations", []):
        img_info = img_by_id.get(ann.get("image_id"))
        if img_info is None:
            continue
        fname = img_info.get("file_name")
        x, y, w, h = ann["bbox"]
        box_xyxy = [x, y, x + w, y + h]
        by_fname.setdefault(fname, []).append({
            "bbox": box_xyxy,
            "category_id": ann.get("category_id", None)
        })
    return by_fname

def match_gt(pred, gts, thr=0.5):
    """
    Elige el GT con mayor IoU respecto a 'pred' si IoU >= thr, si no devuelve None.
    """
    best = None
    best_iou = 0.0
    for g in gts:
        i = iou(pred, g)
        if i > best_iou:
            best, best_iou = g, i
    return best if best_iou >= thr else None

# ======================================================
# LIME
# ======================================================
explainer = lime_image.LimeImageExplainer()
segmenter = SegmentationAlgorithm('slic', n_segments=200)

def lime_wrapper_binary(model, positive_yolo_cls, num_classes=2):
    """
    Función predictiva para LIME (binaria):
      índice 1 = 'humano presente', índice 0 = 'no humano'.
    Usamos p = max(conf de YOLO para la clase positiva en la imagen).
    """
    def f(imgs):
        res = []
        for img in imgs:
            _, scs, cls_ids = predict_yolo(model, img, conf=CONF_THRESH)
            # p = probabilidad de 'humano presente'
            mask = (cls_ids == positive_yolo_cls)
            p = float(np.max(scs[mask])) if mask.any() else 0.0
            p = max(0.0, min(1.0, p))  # limita a [0,1]
            v = np.zeros(num_classes, dtype=float)
            v[1] = p
            v[0] = 1.0 - p
            res.append(v)
        return np.vstack(res)
    return f

# ======================================================
# LOOP PRINCIPAL
# ======================================================
gt_index = load_coco_index(GT_JSON)
print("Ground truth COCO cargado")

results = []

# Ordena para consistencia
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

for img_file in image_files:
    img_path = os.path.join(IMG_DIR, img_file)

    # Carga la imagen en tamaño original (para alinear con GT)
    img = np.array(Image.open(img_path).convert('RGB'))

    # Inferencia YOLO con umbral de detección
    boxes, scores, labels = predict_yolo(model, img, conf=CONF_THRESH)

    # Lista de GTs de esta imagen (sólo "persona" si quieres filtrar por clase)
    gts_all = gt_index.get(img_file, [])
    gts_person = [g["bbox"] for g in gts_all if (g.get("category_id") in (None, HUMAN_CLASS_ID))]

    for i, (b, s, c) in enumerate(zip(boxes, scores, labels)):
        # Solo clase humana y umbral más estricto para explicar
        if c != HUMAN_CLASS_ID:
            continue
        if s < DCL_SCORE_THRESH:
            continue

        # LIME: clase objetivo binaria = índice 1 ('humano')
        f = lime_wrapper_binary(model, HUMAN_CLASS_ID, NUM_CLASSES_LIME)
        explanation = explainer.explain_instance(
            img,
            f,
            top_labels=NUM_CLASSES_LIME,
            num_samples=1000,
            segmentation_fn=segmenter
        )

        target_lime_idx = 1  # 'humano presente'
        sal = np.zeros(img.shape[:2], dtype=float)
        for sp, w in explanation.local_exp[target_lime_idx]:
            sal[explanation.segments == sp] = w
        # Normaliza saliencia a [0,1]
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        # Métricas (búsqueda de la misma clase de YOLO = HUMANO)
        dc = deletion_correlation(img, model, b, HUMAN_CLASS_ID, sal)
        ic = insertion_correlation(img, model, b, HUMAN_CLASS_ID, sal)
        sp_metric = sparsity(sal)

        # Empareja con GT y calcula PG/EBPG
        gt_match = match_gt(b, gts_person, thr=0.5) if len(gts_person) > 0 else None
        pg = pointing_game(sal, gt_match)
        e = ebpg(sal, gt_match)

        results.append({
            "Image": img_file,
            "BBox": i,
            "YOLO_Class": int(c),
            "YOLO_Score": float(s),
            "DC": float(dc),
            "IC": float(ic),
            "Sparsity": float(sp_metric),
            "PG": int(pg),
            "EBPG": float(e),
        })

        # Guardar visualización
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.imshow(sal, cmap='jet', alpha=0.5)
        x1, y1, x2, y2 = b
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_axis_off()

        out_name = f"{os.path.splitext(img_file)[0]}_bb{i}.png"
        plt.savefig(os.path.join(SAL_MAPS_DIR, out_name),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

# ======================================================
# CSV FINAL
# ======================================================
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "metrics_LIME_YOLO_binary.csv")
df.to_csv(csv_path, index=False)
print(f"CSV guardado en: {csv_path}")
print(f"Mapas de saliencia en: {SAL_MAPS_DIR}")