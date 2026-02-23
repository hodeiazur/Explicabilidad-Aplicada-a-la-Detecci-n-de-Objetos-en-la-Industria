import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision import transforms
from drise import DRISEBatch
from tqdm import tqdm
from skimage.segmentation import slic
import pandas as pd
from scipy.stats import pearsonr
import json
import gc

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())

DEVICE = "cuda"
FIXED_SIZE = (640,640) # (W, H) tamaño fijo para D-RISE

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
# YOLO WRAPPER (CLAVE)
# ======================================================
class YOLOWrapper(torch.nn.Module):
    def __init__(self, yolo_model, conf_thres=0.5):
        super().__init__()
        self.yolo = yolo_model
        self.conf = conf_thres
        self.yolo.model.eval() 
    def forward(self, x):
        img = (x[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        results = self.yolo(img, conf=self.conf, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return [{
                "boxes": torch.empty((0,4), device=x.device),
                "scores": torch.empty((0,), device=x.device),
                "labels": torch.empty((0,), dtype=torch.long, device=x.device)
            }]

        boxes = results.boxes.xyxy.to(x.device)
        scores = results.boxes.conf.to(x.device)
        labels = results.boxes.cls.long().to(x.device)

        return [{
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }]

# ======================================================
# PREDICT
# ======================================================
def predict(model, img_rgb):
    img_tensor = torch.from_numpy(
        img_rgb.astype(np.float32)/255.
    ).permute(2,0,1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img_tensor)[0]
    return out

# ======================================================
# MÉTRICAS
# ======================================================
def deletion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, mask_value=0, n_levels=[100,200]):
    H, W = img_rgb.shape[:2]
    segments = slic(img_rgb.astype(np.float32)/255., n_segments=n_levels[0], compactness=10, start_label=0)

    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = img_rgb.copy()
    c_scores, s_scores = [], []

    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = mask_value

        out_pert = predict(model, perturbed)
        best_iou, best_score = 0.0, 0.0

        for b, s, l in zip(out_pert["boxes"].cpu().numpy(),
                           out_pert["scores"].cpu().numpy(),
                           out_pert["labels"].cpu().numpy()):
            if l != orig_class_id:
                continue
            val = iou(orig_box, b)
            if val > best_iou:
                best_iou, best_score = val, s

        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2:
        return 0.0

    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])
    s = np.array(s_scores[:-1])

    if np.std(v) == 0 or np.std(s) == 0:
        return 0.0

    return pearsonr(v, s)[0]

def insertion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, blur_kernel=21, n_levels=[100,200]):
    segments = slic(img_rgb.astype(np.float32)/255., n_segments=n_levels[0], compactness=10, start_label=0)

    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), 0)
    c_scores, s_scores = [], []

    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = img_rgb[seg_mask]

        out_pert = predict(model, perturbed)
        best_iou, best_score = 0.0, 0.0

        for b, s, l in zip(out_pert["boxes"].cpu().numpy(),
                           out_pert["scores"].cpu().numpy(),
                           out_pert["labels"].cpu().numpy()):
            if l != orig_class_id:
                continue
            val = iou(orig_box, b)
            if val > best_iou:
                best_iou, best_score = val, s

        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2:
        return 0.0

    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    s = np.array(s_scores[1:])

    if np.std(v) == 0 or np.std(s) == 0:
        return 0.0

    return pearsonr(v, s)[0]

def sparsity(sal_map):
    return 1.0 / (sal_map.mean() + 1e-8)

def compute_pointing_game(sal_map, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    max_y, max_x = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    return int(x1 <= max_x <= x2 and y1 <= max_y <= y2)

def compute_EBPG(sal_map, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    S = sal_map / (sal_map.sum() + 1e-12)
    return float(S[y1:y2+1, x1:x2+1].sum())

def match_prediction_to_gt(pred_box, gt_boxes, iou_thresh=0.5):
    best_gt, best_iou = None, 0.0
    for gt in gt_boxes:
        val = iou(pred_box, gt)
        if val > best_iou:
            best_iou, best_gt = val, gt
    return best_gt if best_iou >= iou_thresh else None

def load_coco_gt(json_path, image_filename):
    with open(json_path) as f:
        coco = json.load(f)

    img_id = next(img["id"] for img in coco["images"] if img["file_name"] == image_filename)

    gt_boxes, gt_labels = [], []
    for ann in coco["annotations"]:
        if ann["image_id"] == img_id:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([int(x), int(y), int(x+w), int(y+h)])
            gt_labels.append(int(ann["category_id"]))

    return gt_boxes, gt_labels

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = "best.pt"
IMAGE_PATH = "imagenes_testear/rgb_0000.png"
OUTPUT_DIR = "DRISE-METRICAS-rgb_0000-YOLO"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# LOAD MODEL
# ======================================================
yolo = YOLO(MODEL_PATH)
model = YOLOWrapper(yolo, conf_thres=0.5).to(DEVICE)


# ===============================
# PREPROCESAR IMAGEN
# ===============================
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Imagen reducida SOLO para D-RISE
image_small = cv2.resize(image_rgb, FIXED_SIZE)

# Tensores
tensor_full = transforms.ToTensor()(image_rgb).unsqueeze(0).to(DEVICE)
tensor_small = transforms.ToTensor()(image_small).unsqueeze(0).to(DEVICE)

# ======================================================
# BASE PREDICTION
# ======================================================
output = model(tensor_full)[0]
boxes, labels, scores = output["boxes"], output["labels"], output["scores"]

if len(boxes) == 0:
    print("⚠️ No se detectaron objetos.")
    exit()

# ======================================================
# D-RISE
# ======================================================
drise = DRISEBatch(
    model=model,
    input_size=(FIXED_SIZE[0], FIXED_SIZE[1]),
    device=DEVICE,
    N=5000,
    p1=0.25,
    gpu_batch=8
)

drise.generate_masks_rise(N=5000, s=16, p1=0.25)

# ======================================================
# MÉTRICAS
# ======================================================
json_gt_path = "instances_default_rgb_0000.json"
gt_boxes, gt_labels = load_coco_gt(json_gt_path, os.path.basename(IMAGE_PATH))

results = []

for i, (box, cls, conf) in enumerate(zip(boxes, labels, scores)):
    box = box.cpu().numpy()
    cls = int(cls.item())
    conf = float(conf.item())
    image_bbox = image_small.copy()

    # Escalar bbox al tamaño FIXED_SIZE
    H0, W0 = image_rgb.shape[:2]
    sx, sy = FIXED_SIZE[0] / W0, FIXED_SIZE[1] / H0

    box_small = [
        box[0] * sx, box[1] * sy,
        box[2] * sx, box[3] * sy
    ]

    target_bbox = [
        (box_small[0], box_small[1]),
        (box_small[2], box_small[1]),
        (box_small[2], box_small[3]),
        (box_small[0], box_small[3]),
    ]

    saliency_maps = drise.forward(
        x=tensor_small,
        target_class_indices=[cls],
        target_bbox=target_bbox
    )

    saliency_small = saliency_maps[cls]
    saliency = cv2.resize(
        saliency_small,
        (W0, H0),
        interpolation=cv2.INTER_LINEAR
    )
    sal_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-12)
    # ===============================
    # Guardar imagen con heatmap
    # ===============================
    h, w = image.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    heatmap = cv2.applyColorMap(
    (sal_norm * 255).astype(np.uint8),cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    cv2.rectangle(overlay, (x1, y1), (x2, y2),(0, 255, 0), 3)

    cv2.putText(
        overlay,
        f"class {cls} ({conf:.2f})",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"object_{i+1}_class_{cls}.png"),overlay)

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
