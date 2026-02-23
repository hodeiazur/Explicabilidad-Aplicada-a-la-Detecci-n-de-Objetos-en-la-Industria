import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from drise import DRISEBatch  
from tqdm import tqdm
from skimage.segmentation import slic
import pandas as pd
DEVICE = 'cpu'
# torch.cuda.empty_cache()  # libera memoria no usada reservada por PyTorch
# print(torch.cuda.memory_summary())
# ======================================================
#  Funciones auxiliares
# ======================================================
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)

def predict(model, img_rgb):
    """Inferencia y salida en CPU"""
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_tensor)[0]
    return out
# ======================================================
# Métricas
# ======================================================
from scipy.stats import pearsonr

def deletion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, mask_value=0, n_levels=[100,200]):
    H, W = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=n_levels[0], compactness=10, start_label=0)
    
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]  # más importante primero
    
    perturbed = img_rgb.copy()
    c_scores = []
    s_scores = []
    
    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = mask_value
        out_pert = predict(model, perturbed)
        
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out_pert["boxes"].cpu().numpy(), out_pert["scores"].cpu().numpy(), out_pert["labels"].cpu().numpy()):
            if l != orig_class_id: continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)
    
    if len(c_scores) < 2:
        return 0.0  # no hay suficientes pasos para correlación

    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])
    s = np.array(s_scores[:-1])
    
    if np.std(v) == 0 or np.std(s) == 0:
        return 0.0  # evitar pearsonr con constante
    corr, _ = pearsonr(v, s)
    return corr

def insertion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, blur_kernel=21, n_levels=[100,200]):
    H, W = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=n_levels[0], compactness=10, start_label=0)
    
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]  # más importante primero

    perturbed = cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), 0)
    c_scores = []
    s_scores = []

    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = img_rgb[seg_mask]
        
        out_pert = predict(model, perturbed)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out_pert["boxes"].cpu().numpy(), out_pert["scores"].cpu().numpy(), out_pert["labels"].cpu().numpy()):
            if l != orig_class_id: continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)
    
    if len(c_scores) < 2:
        return 0.0

    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    s = np.array(s_scores[1:])
    
    if np.std(v) == 0 or np.std(s) == 0:
        return 0.0
    corr, _ = pearsonr(v, s)
    return corr

def sparsity(sal_map): #Smax/Smin == 1/mean(S) (porque está normalizado)
    # sal_map ya normalizado a [0,1]
    return 1.0 / (sal_map.mean() + 1e-8)


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

def match_prediction_to_gt(pred_box, gt_boxes, iou_thresh=0.5):
    best_gt = None
    best_iou_val = 0.0

    for gt in gt_boxes:
        val = iou(pred_box, gt)
        if val > best_iou_val:
            best_iou_val = val
            best_gt = gt

    if best_gt is None or best_iou_val < iou_thresh:
        return None
    return best_gt

import json

#obtener el gt real de la imagen del testeo para poder calcular la métrica
def load_coco_gt(json_path, image_filename):
    """
    Devuelve:
      gt_boxes  → lista [[x1,y1,x2,y2], ...]
      gt_labels → lista [category_id, ...]
    A partir de un .json COCO como el que has mostrado.
    """
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # === 1) Buscar la imagen por nombre ===
    img_id = None
    for img in coco["images"]:
        if img["file_name"] == image_filename:
            img_id = img["id"]
            break

    if img_id is None:
        raise ValueError(f"No se ha encontrado la imagen '{image_filename}' en el JSON COCO.")

    # === 2) Recoger todas las anotaciones para esa imagen ===
    gt_boxes = []
    gt_labels = []

    for ann in coco["annotations"]:
        if ann["image_id"] == img_id:
            x, y, w, h = ann["bbox"]
            x1, y1 = x, y
            x2, y2 = x + w, y + h  # Convertir COCO bbox -> formato normal

            gt_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            gt_labels.append(int(ann["category_id"]))

    return gt_boxes, gt_labels


# ===============================
# CONFIGURACIÓN
# ===============================


MODEL_PATH = "fasterrcnn_trained_4.pth"
IMAGE_PATH = "imagenes_testear/00901.jpg"
OUTPUT_DIR = "DRISE-METRICAS-00901"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# CARGAR MODELO
# ===============================
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=12)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

# ===============================
# PREPROCESAR IMAGEN
# ===============================
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = transforms.Compose([transforms.ToTensor()])
tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

# ===============================
# PREDICCIÓN BASE
# ===============================
with torch.no_grad():
    output = model(tensor)[0]

boxes = output['boxes']
labels = output['labels']
scores = output['scores']

conf_thres = 0.5
mask = scores > conf_thres 
boxes, labels, scores = boxes[mask], labels[mask], scores[mask]

if len(boxes) == 0:
    print("⚠️ No se detectaron objetos con confianza suficiente.")
    exit()

print(f"Se detectaron {len(boxes)} objetos con confianza > {conf_thres}")

# ===============================
# INSTANCIAR D-RISE UNA SOLA VEZ
# ===============================
drise = DRISEBatch(
    model=model,
    input_size=(image.shape[0], image.shape[1]),  # alto, ancho
    device=DEVICE,
    N=500,        # número de máscaras
    p1=0.25,       # proporción visible
    gpu_batch=50  # tamaño de batch en GPU
)

# drise.generate_masks_rise(N=1000, s=6, p1=0.2)

# ===============================
# MÉTRICAS PARA CADA DETECCIÓN
# ===============================
json_gt_path = "instances_default_00901.json"
gt_boxes, gt_labels = load_coco_gt(json_gt_path, os.path.basename(IMAGE_PATH))

results = []

for i, (box, cls, conf) in enumerate(zip(boxes, labels, scores)):
    box = box.cpu().numpy()
    cls = int(cls.item())
    conf = float(conf.item())
    target_bbox = [(box[0], box[1]), (box[2], box[1]), (box[2], box[2]), (box[0], box[2])]
    target_classes = [cls]

    saliency_maps = drise.forward(x=tensor, target_class_indices=target_classes, target_bbox=target_bbox)
    saliency = saliency_maps[cls]
    saliency_resized = cv2.resize(saliency, (image.shape[1], image.shape[0]))
    sal_norm = (saliency_resized - saliency_resized.min()) / (saliency_resized.max() - saliency_resized.min() + 1e-12)

    # ===============================
    # Guardar imagen con heatmap
    # ===============================
    heatmap = cv2.applyColorMap((sal_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    heatmap_filename = os.path.join(OUTPUT_DIR, f"object_{i+1}_class_{cls}.png")
    cv2.imwrite(heatmap_filename, overlay)

    # ===============================
    # Sparsity
    # ===============================
    sp = sparsity(sal_norm)

    # ===============================
    # Pointing Game
    # ===============================
    gt_match = match_prediction_to_gt(box, gt_boxes)
    pg_hit = compute_pointing_game(sal_norm, gt_match) if gt_match is not None else 0

    # ===============================
    # EBPG
    # ===============================
    ebpg_score = compute_EBPG(sal_norm, gt_match) if gt_match is not None else 0

    # ===============================
    # DC / IC
    # ===============================
    dc_score = deletion_correlation(image_rgb, model, box, cls, conf, sal_norm)
    ic_score = insertion_correlation(image_rgb, model, box, cls, conf, sal_norm)

    # ===============================
    # Guardar resultados
    # ===============================
    results.append({
        "Object": i+1,
        "Class": cls,
        "Score": conf,
        "DC": dc_score,
        "IC": ic_score,
        "Sparsity": sp,
        "PG_Hit": pg_hit,
        "EBPG": ebpg_score
    })

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "metrics_2.csv"), index=False)

print("Métricas guardadas en CSV y heatmaps generados.")
