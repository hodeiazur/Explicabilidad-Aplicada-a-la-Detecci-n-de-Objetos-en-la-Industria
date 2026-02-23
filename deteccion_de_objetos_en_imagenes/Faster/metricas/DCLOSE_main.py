"""El D-CLOSE global real siempre va a ser lento con Faster R-CNN si se hace pixel/superpíxel por pixel/superpíxel, porque hace cientos de inferencias pesadas."""

import os
import cv2
import numpy as np
import torch
import torchvision
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd


# ======================================================
#  Cargar modelo Faster R-CNN entrenado
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
num_classes = 12
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
model.load_state_dict(torch.load("fasterrcnn_trained_4.pth", map_location=device))
model.to(device).eval()

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
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)[0]
    return out

# ======================================================
# D-CLOSE global (perturbaciones)
# ======================================================
def dclose_map_global(img_rgb, model, orig_box, orig_class_id, orig_score,
                      n_levels=[100,200], mask_value=0, iou_thresh=0.3):

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
            boxes_p = out_pert["boxes"].cpu().numpy()
            scores_p = out_pert["scores"].cpu().numpy()
            labels_p = out_pert["labels"].cpu().numpy().astype(int)

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

def sparsity(sal_map): #Smax/Smin == 1/mean(S) (porque está nromalizado)
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



# ======================================================
# Uso
# ======================================================
img_path = "imagenes_testear/rgb_0000.png"
json_path = "imagenes_testear/instances_default_rgb_0000.json"

gt_boxes, gt_labels = load_coco_gt(json_path, img_path)

print("GT boxes:", gt_boxes)
print("GT labels:", gt_labels)

output_dir = "DCLOSE-METRICAS-RGB_0000"
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

out = predict(model, img_rgb)
boxes = out["boxes"].cpu().numpy().astype(int)
scores = out["scores"].cpu().numpy()
classes = out["labels"].cpu().numpy().astype(int)

colors = {
    0:(255,0,0),1:(0,255,0),2:(0,0,255),3:(255,255,0),
    4:(255,0,255),5:(0,255,255)
}


# Crear lista para acumular resultados
results = []

for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
    if score < 0.5:
        continue

    # --- Saliency map de D-CLOSE ---
    sal_map = dclose_map_global(img_rgb, model, box, cls, score, n_levels=[100,200])
    
    # --- Localización ---
    dc = deletion_correlation(img_rgb, model, box, cls, score, sal_map)
    ic = insertion_correlation(img_rgb, model, box, cls, score, sal_map)
    sp = sparsity(sal_map)
    
    # --- Emparejar con GT --- Fidelidad --- 
    gt_box = match_prediction_to_gt(box, gt_boxes)

    if gt_box is not None:
        pg  = compute_pointing_game(sal_map, gt_box)
        ebpg = compute_EBPG(sal_map, gt_box)
    else:
        pg = 0
        ebpg = 0.0

    # --- Guardar en lista ---
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

    # ---- Visualización ----
    x1, y1, x2, y2 = box
    img_show = img_rgb.copy()
    color = colors.get(cls,(255,255,255))
    cv2.rectangle(img_show,(x1,y1),(x2,y2), color,2)
    cv2.putText(img_show,f"cls {cls}: {score:.2f}",
                (x1, max(10,y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.figure(figsize=(6,6))
    plt.imshow(img_show)
    plt.imshow(sal_map, cmap='jet', alpha=0.5)
    plt.axis('off')

    save_path = os.path.join(output_dir, f"obj{i+1}_cls{cls}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# --- Guardar CSV ---
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "metrics_completo.csv")
df.to_csv(csv_path, index=False)
print(f"CSV guardado en {csv_path}")


"""D-CLOSE puede ser más rápido que D-RISE si reduces el número de superpíxeles (por ejemplo 50-200).

D-RISE generalmente es más tardado, porque suele usar 1000+ máscaras.


Sin optimizaciones de batch, ambos métodos son lentos en Faster R-CNN, pero D-CLOSE es más controlable (puedes ajustar n_levels y compactness)."""
